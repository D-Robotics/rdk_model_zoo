/**
 * @file paraformer.hpp
 * @brief Paraformer UCP evaluator entry point.
 *
 * The implementation keeps the validated Encoder -> Predictor -> CPU CIF ->
 * Decoder pipeline used by the Paraformer S100 deployment.
 */

#ifndef PARAFORMER_HPP_
#define PARAFORMER_HPP_

/**
 * @brief Run the Paraformer manifest evaluator.
 *
 * @param[in] argc Number of command-line arguments.
 * @param[in] argv Command-line argument values.
 * @return Exit status; zero indicates success.
 */
int paraformer_main(int argc, char** argv);

#endif  // PARAFORMER_HPP_
