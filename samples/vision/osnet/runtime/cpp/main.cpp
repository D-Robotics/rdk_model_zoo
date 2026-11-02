#include <cstring>
int osnet_main(int argc, char** argv);
int osnet_benchmark_main(int argc, char** argv);
int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--benchmark") == 0)
        return osnet_benchmark_main(argc - 1, argv + 1);
    return osnet_main(argc, argv);
}
