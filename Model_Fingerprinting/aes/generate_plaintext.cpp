#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <fstream>

std::string create_input(size_t len)
{
	const std::string alphanum = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz !@#$%%^&*()~{}:\"?><][';/.,|`";
	std::string str;
	int pos;

	while(str.size() != len) {
		pos = rand() % (alphanum.size() - 1);
		str += alphanum.substr(pos,1);
	}
	return str;
}

int main(int argc, char* argv[]) {

  if (argc < 5) {
    std::cout << "usage: " << argv[0] << " <mean> <stddev> <Num_lines> <outfile>\n    mean & stddev normal dist for input size. Outfile is output txt file to dump plaintext into.\n";
    exit(1);
  }

  int totalSize = 0;
  srand ( time(NULL) );
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<float> distribution(std::stof(argv[1]), std::stof(argv[2]));
  int N = distribution(generator);
  int numLines = std::stoi(argv[3]);
  if (numLines < 0) {
    std::cerr << "num lines cannot be less than 0" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string filename = argv[4];

  std::fstream out_file;
  out_file.open(filename, std::ios::out); 

  if (!out_file.is_open()) {
    std::cerr << "Cannot open input file: " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  // for (int i = 0; i < numLines; i++) {
  //   // std::cout << "Input size: " << N << "\n";
  //   out_file << create_input(N);
  //   out_file << "\n";
  //   totalSize += N;
  //   N = distribution(generator);
  // }
  std::cout << "Input size: " << N << "\n";
  out_file << create_input(N);

  if (out_file.is_open()) {
    out_file.close();
  }

  std::cout << "Total size: " << totalSize + numLines << " (Bytes)"<< "\n";

  return 0;
}