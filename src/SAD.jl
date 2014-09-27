module SAD
using GaussianMixtureModels, Features, Stage, Iterators
import Base: start, done, next, length
export select_frames, rats_speech, rats_nonspeech, speech, nonspeech, split_train, kmeans_train, score, test, optimize

# -------------------------------------------------------------------------------------------------------------------------
# Types
# -------------------------------------------------------------------------------------------------------------------------
immutable LazyMap{I}
  flt::Function
  itr::I
end
lazy_map(f::Function, itr) = LazyMap(f, itr)

function start(m :: LazyMap) 
  s = start(m.itr)
  return s
end

function next(m :: LazyMap, s) 
  n, ns = next(m.itr, s)
  return (m.flt(n), ns)
end

done(m :: LazyMap, s) = done(m.itr, s)

# -------------------------------------------------------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------------------------------------------------------
function select_frames(sf :: SegmentedFile; masker = sf -> mask(sf))
  m, n = masker(sf)
  lazy_map(x -> x[2], filter(f -> m[f[1]], enumerate(HTKFeatures(sf.fn))))
end

# RAT's style explicit speech/nonspeech labels
rats_speech(sf :: SegmentedFile)    = select_frames(sf, masker = sf -> mask(sf, filter = (kind, start, fin, file) -> kind == "S"))
rats_nonspeech(sf :: SegmentedFile) = select_frames(sf, masker = sf -> mask(sf, filter = (kind, start, fin, file) -> kind == "NS" || kind == "NT"))

# analist-style speech labels
function negate_mask(sf :: SegmentedFile)
  m, n = mask(sf)
  res = [!x for x in m]
  return res, count(x -> x, res)
end

speech(sf :: SegmentedFile)    = select_frames(sf)
nonspeech(sf :: SegmentedFile) = select_frames(sf, masker = negate_mask)

# -------------------------------------------------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------------------------------------------------
function split_train(files; splits = 6, iterations = 5, speech_frames = sf -> speech(sf), nonspeech_frames = sf -> nonspeech(sf))
  s     = map(speech_frames, files)
  ns    = map(nonspeech_frames, files)
  
  d             = dims(HTKFeatures(files[1].fn))
  speech_gmm    = GMM(d, 1)
  nonspeech_gmm = GMM(d, 1)

  for i = 1:(splits + 1)
    @timer "training $(2^(i-1))g speech models"    new_s  = par_train(speech_gmm, s, iterations = iterations)
    @timer "training $(2^(i-1))g nonspeech models" new_ns = par_train(nonspeech_gmm, ns, iterations = iterations)
    if i < (splits + 1)
      speech_gmm    = split(new_s)
      nonspeech_gmm = split(new_ns)
    end
  end

  return speech_gmm, nonspeech_gmm
end

function take_all(iter, size)
  buffer = Array(Any, size)
  idx = 1
  for i in iter
    if idx <= size
      buffer[idx] = i
      idx += 1
    end
  end
  buffer
end

function kmeans_train(files; g = 64, iterations = 5, sample_size = 1500, speech_frames = sf -> speech(sf), nonspeech_frames = sf -> nonspeech(sf))
  init_s    = map(sf -> speech_frames(sf), files[sortperm(files, by = x -> rand())[1:min(sample_size, end)]])
  init_ns   = map(sf -> nonspeech_frames(sf), files[sortperm(files, by = x -> rand())[1:min(sample_size, end)]])
  speech    = map(speech_frames, files)
  nonspeech = map(nonspeech_frames, files)
  
  d             = dims(HTKFeatures(files[1].fn))
  speech_gmm    = @spawn kmeans_init(GMM(d, g), chain(init_s...))
  nonspeech_gmm = @spawn kmeans_init(GMM(d, g), chain(init_ns...))
  
  @timer "$(g)g training of speech models"     par_train(fetch(speech_gmm), speech, iterations = iterations)
  @timer "$(g)g training of non-speech models" par_train(fetch(nonspeech_gmm), nonspeech, iterations = iterations)

  return fetch(speech_gmm), fetch(nonspeech_gmm)
end

# -------------------------------------------------------------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------------------------------------------------------------
function score(sf :: SegmentedFile, speech :: GMM, nonspeech :: GMM; window_radius = 10)
  scores = Float32[]
  for (i, f) in enumerate(HTKFeatures(sf.fn))
    scr = ll(speech, f) - ll(nonspeech, f)
    push!(scores, scr)
  end

  # smooth
  smoothed = zeros(length(scores))
  for s = 1:length(scores)
    start = max(1, s-window_radius)
    fin   = min(length(scores), s+window_radius)
    total = 0.0
    for i = start:fin
      total += scores[i]
    end
    smoothed[s] = total / (fin - start + 1)
  end

  return smoothed
end

function filter_short_segments(segs :: Vector{(Int, Int)}; min_duration = 20, verbose = false)
  nonshort_segs = (Int, Int)[]
  for i = 1:length(segs)
    s, e = segs[i]
    if (e - s + 1) < min_duration
      verbose && @info "removing segment from $s to $e: too short"
    else
      push!(nonshort_segs, (s, e))
    end
  end
  
  return nonshort_segs
end

function degap_segments(segs :: Vector{(Int, Int)}; min_gap = 15, verbose = false)
  degapped_segs = (Int, Int)[]
  for i = 1:length(segs)
    s, e = segs[i]
    if length(degapped_segs) > 1 && (s - degapped_segs[end][2]) < min_gap
      verbose && @info "combining segment [$(degapped_segs[end][1]) to $(degapped_segs[end][2])] with [$s to $e]: short gap"
      degapped_segs[end] = (degapped_segs[end][1], e)
    else
      push!(degapped_segs, (s, e))
    end
  end

  return degapped_segs
end

function filter_segments(scores; threshold = 0.0, min_duration = 15, min_gap = 10, verbose = false)
  decs = [ s >= threshold for s in scores ]
  segs = (Int, Int)[]
  for i = 1:length(decs)
    start = decs[i] && (i == 1 || (i > 1 && decs[i - 1] == false))
    if start
      j = i
      while (j < length(decs)) && decs[j]
        j += 1
      end
      push!(segs, (i, j))
    end
  end

  filtered = degap_segments(filter_short_segments(segs, min_duration = min_duration, verbose = verbose), min_gap = min_gap, verbose = verbose)
  ret = [ false for s in scores ]
  for i = 1:length(filtered)
    s, e = filtered[i]
    for j = s:e
      ret[j] = true
    end
  end

  return ret
end

function test(files, speech :: GMM, nonspeech :: GMM; verbose = false, min_gap = 10, min_duration = 20, threshold = 0.0, speech_mask = sf -> mask(sf), nonspeech_mask = sf -> negate_mask(sf))
  decs   = pmap(f -> filter_segments(score(f, speech, nonspeech), threshold = threshold, min_gap = min_gap, min_duration = min_duration, verbose = verbose), files) 
  #[ filter_segments(score(f, speech, nonspeech), threshold = threshold, min_gap = min_gap, min_duration = min_duration, verbose = verbose) for f in files ]
  N      = 0
  FAs    = 0
  misses = 0
  
  for (k, sf) in enumerate(files)
    s_mask, s_frames   = speech_mask(sf)
    ns_mask, ns_frames = nonspeech_mask(sf)
    for i = 1:length(s_mask)
      dec = decs[k][i]
      if !dec && (s_mask[i] && !ns_mask[i])
        misses += 1
      elseif dec && (ns_mask[i] && !s_mask[i])
        FAs += 1
      end
      if s_mask[i] || ns_mask[i]
        N += 1
      end
    end
  end

  return FAs, misses, N, decs
end

function optimize(files, speech, nonspeech; c_miss = 1.0, c_fa = 1.0, speech_mask = sf -> mask(sf), nonspeech_mask = sf -> negate_mask(sf))
  scores     = @parallel (vcat) for f in files score(f, speech, nonspeech) end #[ score(f, speech, nonspeech) for f in files ]
  truth      = Int8[]
  flatscores = Float32[]

  # gather scores
  for i = 1:length(scores)
    for s in scores[i]
      push!(flatscores, s)
    end
  end

  # gather truth
  N_speech = 0
  N        = 0
  N_ns     = 0
  for sf in files
    s_mask, s_frames   = speech_mask(sf)
    ns_mask, ns_frames = nonspeech_mask(sf)
    for i = 1:length(s_mask)
      if s_mask[i]
        push!(truth, 1)
        N_speech += 1
        N        += 1
      elseif ns_mask[i]
        push!(truth, 0)
        N_ns += 1
        N    += 1
      else
        push!(truth, 2)
      end
    end
  end

  # sweep
  dc   = zeros(length(flatscores))
  fa   = zeros(length(flatscores))
  miss = zeros(length(flatscores))
  hits = 0
  fas  = 0
  eer  = 0.0
  eerd = 10000.0

  indexes = sortperm(flatscores, rev = true)
  for idx in indexes
    scr     = flatscores[idx]
    speechp = truth[idx]
    if speechp == 1
      hits += 1
    elseif speechp == 0
      fas += 1
    end

    far       = fas / float32(N_ns)
    missr     = (N_speech - hits) / float32(N_speech)
    fa[idx]   = far
    miss[idx] = missr
    dc[idx]   = far * c_fa + missr * c_miss

    if eerd > abs(fa[idx] - miss[idx])
      eerd = abs(fa[idx] - miss[idx])
      eer  = (fa[idx] + miss[idx]) / 2.0
    end
  end

  best = indmin(dc)
  return flatscores[best], fa[best], miss[best], dc[best], fa, miss, eer, N_speech, N_ns
end

end # module
