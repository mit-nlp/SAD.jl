using SAD, Features, DocOpt, Stage

S(sf)  = mask(sf, filter = (kind, start, fin, file) -> kind == "S")
NS(sf) = mask(sf, filter = (kind, start, fin, file) -> kind == "NS" || kind == "NT")

# ----------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------

usage = """RATS SAD Train/test script
Usage:
  rats.jl [options]

Options:
  --audio=dir, -a     Audio Directory relative to current path [default: rats-sample]
  --test=M            Marks for testing and threshold optimization
  --train=M           Marks with marks for training
  --kmeans=s          Do K-Means Initialization instead of binary splitting the argument is the sample size [default: 0]
  --model=s           Name of output model [default: rats-sad.gmm]
  --iterations=i      Number of EM training iterations to perform during GMM training [default: 5]
  --gaussians=g, -g   Number of gaussians to target for final GMM (should be a power of two if binary splitting [default: 2]
"""

args = docopt(usage, ARGS, version=v"0.0.1")

# train
if args["--train"] != nothing
  @info "training with $(args["--train"]) using audio from $(args["--audio"]) with $(args["--gaussians"]) gaussians and $(args["--iterations"]) iterations"
  files = marks(args["--train"], dir = args["--audio"])
  s, ns = int(args["--kmeans"]) > 0 ? kmeans_train(files, g = int(args["--gaussians"]), iterations = int(args["--iterations"]), sample_size = int(args["--kmeans"]),
                                                   speech_frames = rats_speech, nonspeech_frames = rats_nonspeech) :
                                      split_train(files, splits = int(log2(int(args["--gaussians"]))), iterations = int(args["--iterations"]), 
                                                  speech_frames = rats_speech, nonspeech_frames = rats_nonspeech)
  outf = open(args["--model"], "w")
  serialize(outf, s)
  serialize(outf, ns)
  close(outf)
end

# score
if args["--test"] != nothing
  f  = open(args["--model"], "r")
  s  = deserialize(f)
  ns = deserialize(f)
  close(f)

  files = marks(args["--test"], dir = args["--audio"], collars = { "S" => 0.25f0, "NS" => 0.1f0, "NT" => 0.1f0 })
  threshold, opt_fa, opt_miss, opt_dc, fa, miss, eer, N_speech, N_ns = optimize(files, s, ns, speech_mask = S, nonspeech_mask = NS)
  @info "optimum threshold = $threshold [eer = $eer, fa rate = $opt_fa, miss rate = $opt_miss, decision cost = $opt_dc]"

  FAs, misses, N, decisions = test(files, s, ns, threshold = threshold, speech_mask = S, nonspeech_mask = NS)
  @info "FA rate   = $(FAs / N_ns) (N = $FAs / $N_ns)"
  @info "Miss rate = $(misses / N_speech) (N = $misses / $N_speech)"
  @info "N         = $N"
end
