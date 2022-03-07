# AES

------------------------------
Install cryptopp
------------------------------

	cd cryptopp
	make static cryptest.exe
	./cryptest.exe v
	./cryptest.exe tv
	sudo make install PREFIX=/usr/local

------------------------------
Compile Example File
------------------------------

	g++ -o aesExampleBin aesExample.cpp -lcryptopp

------------------------------
Generate input plaintext files
------------------------------

	generate_plaintext.cpp

------------------------------
Source
------------------------------
Code has been adapted from:
[HERMIT Benchmark Suite](https://github.com/ankurlimaye/HERMIT-BenchmarkSuite)

