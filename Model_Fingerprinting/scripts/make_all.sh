#!/bin/bash

# Usage ./make_all.sh
# NOTE: bc needs to be installed to do runtime calc arithmetic

if [ $# -eq 0 ]; then
	echo "Usage: ./make_all.sh <Num of runs for average>"
	exit 1
fi

# Number of runs to get average runtime
N=$1

# List of programs to collect data from
declare -a programs=("squeezenet1_0" "squeezenet1_1" "resnet18" "resnet50" "resnet152" "densenet121" "densenet169" 
"densenet201" "vgg11" "vgg13" "vgg19" "alexnet" "inceptionv3" "mobilenetv2")


# Command
cmd=(python3)
averages=()

# Collect N traces from each program
for i in "${programs[@]}"; do
	run_cmd=${cmd[@]}
	run_cmd+=(../models/${i}.py)
	
	echo "########## ${i} ##########"
	runtimes=()
	for j in $(seq 1 $N); do
		# Run & time program
		start=$(date +%s.%N)
		${run_cmd[@]} ../imagenet/
		end=$(date +%s.%N)
		runtime=$( echo "$end - $start" | bc -l )
		runtimes+=( "$runtime" )
		echo "Run #${j}: $runtime (s)"
		sleep 2
	done

	# Get average runtime for current model
	sum=0
	m=$N
	while [ $m -gt 0 ]; do
		# copy elem in a temp
		temp=${runtimes[`expr $m - 1`]}
		# add to sum
		sum=`echo "$sum + $temp" | bc -l`
		# decrement m
		m=`expr $m - 1`
	done
	# Compute Average runtime
	avg=`echo "$sum / $N" | bc -l`
	echo "Avg Runtime: $avg (s)"
	echo "---------------------"
	
	averages+= ( "$avg" )
	unset run_cmd[-1]

done

echo -e "\n########### Average Runtimes ###########\n"
count=0
for i in "${programs[@]}"; do
	echo "${i} -- ${averages[$count]} (s)"
	let count++
done
