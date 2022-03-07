#!/usr/bin/perl
use strict;
use warnings;
use Cwd;

my $cwd = getcwd();
my $bench;

my $dirFolder;
my $dir;
my $newDir;

my $rVar = "500";
my $runFile;

### network > dijkstra > dijkstra_small

$dirFolder = "";
$dir = "aes";
$newDir = "$cwd/$dirFolder/$dir";
#chdir ($newDir);
chdir($cwd);

$bench = 'perfOut_'.$dir.'_'.$rVar.'x_4flags.txt';
$runFile = "outputOfExample";
#$runFile = "runme_large.sh";

print "Now running: $dirFolder > $dir > $runFile \n";
system("perf stat -r $rVar -e branch-instructions,branch-misses,cache-references,cache-misses ./$runFile 2> $bench");
system("perf stat -r $rVar -e bus-cycles,cpu-cycles,instructions ./$runFile 2>> $bench");
system("perf stat -r $rVar -e alignment-faults,bpf-output,dummy ./$runFile 2>> $bench");
system("perf stat -r $rVar -e cpu-clock,task-clock,cpu-migrations,context-switches ./$runFile 2>> $bench");
system("perf stat -r $rVar -e emulation-faults,major-faults,minor-faults,page-faults ./$runFile 2>> $bench");
system("perf stat -r $rVar -e L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses ./$runFile 2>> $bench");
system("perf stat -r $rVar -e L1-icache-loads,L1-icache-load-misses,branch-loads,branch-load-misses ./$runFile 2>> $bench");
system("perf stat -r $rVar -e LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses ./$runFile 2>> $bench");
system("perf stat -r $rVar -e dTLB-load-misses,dTLB-store-misses,iTLB-load-misses ./$runFile 2>> $bench");

