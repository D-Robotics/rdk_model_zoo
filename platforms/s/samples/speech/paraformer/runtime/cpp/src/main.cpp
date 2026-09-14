/**
 * @file main.cpp
 * @brief Standard executable entry point for the Paraformer sample.
 */

#include "paraformer.hpp"

/**
 * @brief Delegate execution to the Paraformer pipeline.
 *
 * @param[in] argc Number of command-line arguments.
 * @param[in] argv Command-line argument values.
 * @return Exit status from the Paraformer evaluator.
 */
int main(int argc, char** argv)
{
    return paraformer_main(argc, argv);
}
