#!/usr/bin/env bash
#
nj=4
cmd=run.pl
compress=true
write_utt2num_frames=true

if [ -f kaldi_path.sh ]; then . ./kaldi_path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
  cat >&2 <<EOF
Usage: $0 [options] <feats-in-dir> [<log-dir> [<feats-out-dir>] ]
 e.g.: $0 data/train
Note: <log-dir> defaults to <feats-in-dir>/log, and
      <feats-out-dir> defaults to <feats-out-dir>/ark.
Options:
  --nj <nj>                            # number of parallel jobs.
  --cmd <run.pl|queue.pl <queue opts>> # how to run jobs.
  --write-utt2num-frames <true|false>  # If true, write utt2num_frames file.
EOF
   exit 1;
fi

data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=$data/log
fi
if [ $# -ge 3 ]; then
  featsdir=$3
else
  featsdir=$data/ark
fi

if [ ! -d $featsdir ]; then
    mkdir -p $featsdir || exit 1;
fi

if [ ! -d $logdir ]; then
    mkdir -p $logdir || exit 1;
fi


if $write_utt2num_frames; then
  write_num_frames_opt="--write-num-frames=ark,t:$logdir/speech_shape.JOB"
else
  write_num_frames_opt=
fi


$cmd JOB=1:$nj $logdir/make_feats_scp.JOB.log \
    copy-feats --compress=$compress $write_num_frames_opt ark:${data}/ark/fbank.JOB.ark \
      ark,scp:$featsdir/fbank.JOB.ark,$featsdir/fbank.JOB.scp \
     || exit 1;

$cmd JOB=1:$nj $logdir/make_target_scp.JOB.log \
    copy-int-vector ark:${data}/target/post.JOB.ark.txt \
      ark,scp:$featsdir/target.JOB.ark,$featsdir/target.JOB.scp \
     || exit 1;

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $featsdir/fbank.$n.scp || exit 1
done > $featsdir/feats.scp || exit 1

for n in $(seq $nj); do
  cat $featsdir/target.$n.scp || exit 1
done > $featsdir/target.scp || exit 1

if $write_utt2num_frames; then
  for n in $(seq $nj); do
    cat $logdir/speech_shape.$n || exit 1
  done > $featsdir/speech_shape || exit 1
fi

cp $featsdir/speech_shape $featsdir/text_shape
