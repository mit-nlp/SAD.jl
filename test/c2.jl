using SAD, Features, DocOpt, Stage

# ----------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------

usage = """C2 SAD Train/test script
Usage:
  rats.jl [options]

Options:
  --audio=dir, -a    Audio Directory relative to current path [default: rats-sample]
  --test=M           Marks for testing and threshold optimization
  --train=M          Marks with marks for training
  --kmeans           Do K-Means Initialization instead of binary splitting [default: false]
  --model=s          Name of output model [default: rats-sad.gmm]
  --iterations=i     Number of EM training iterations to perform during GMM training [default: 5]
  --gaussians=g, -g  Number of gaussians to target for final GMM (should be a power of two if binary splitting [default: 2]         
"""

args = docopt(usage, ARGS, version=v"0.0.1")

# train
if args["--train"] != nothing
  @info "training with $(args["--train"]) using audio from $(args["--audio"]) with $(args["--gaussians"]) gaussians and $(args["--iterations"]) iterations"
  files = analist(args["--train"], dir = args["--audio"])
  speech, nonspeech = args["--kmeans"] ? kmeans_train(files, g = int(args["--gaussians"]), iterations = int(args["--iterations"]), sample_size = 700) :
                                         split_train(files, splits = int(log2(int(args["--gaussians"]))), iterations = int(args["--iterations"]))
  outf = open(args["--model"], "w")
  serialize(outf, speech)
  serialize(outf, nonspeech)
  close(outf)
end

# score
if args["--test"] != nothing
  f = open(args["--model"], "r")
  speech    = deserialize(f)
  nonspeech = deserialize(f)
  close(f)

  files = analist(args["--test"], dir = args["--audio"])
  threshold, opt_fa, opt_miss, opt_dc, fa, miss, eer, N_speech, N_ns = optimize(files, speech, nonspeech)
  @info "optimum threshold = $threshold [eer = $eer, fa rate = $opt_fa, miss rate = $opt_miss, decision cost = $opt_dc]"

  FAs, misses, N, decisions = test(files, speech, nonspeech, threshold = threshold)
  @info "FA rate   = $(FAs / N_ns) (N = $FAs / $N_ns)"
  @info "Miss rate = $(misses / N_speech) (N = $misses / $N_speech)"
  @info "N         = $N"
end
