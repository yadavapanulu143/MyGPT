// Minimal greedy-decoding harness used to prove that the GGUF export of MyGPT
// reproduces the PyTorch reference model token-for-token.
//
// Build (from the llama.cpp checkout, after building libllama):
//   g++ -O2 -std=c++17 -I $LLAMA_DIR/include -I $LLAMA_DIR/ggml/include \
//       tools/llama_parity.cpp -L $LLAMA_DIR/build/bin \
//       -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA_DIR/build/bin -o /tmp/llama_parity
//
// Usage: llama_parity model.gguf "prompt text" 12 [n_ctx]
// Output: one JSON object with prompt token ids and greedily generated ids.
// Weights are read into anonymous memory (no mmap) so RSS is a true measure of
// what the process needs, not of how much of the file the kernel has cached.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf \"prompt\" n_predict\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string prompt = argv[2];
    const int n_predict = atoi(argv[3]);
    const int n_ctx = argc > 4 ? atoi(argv[4]) : 256;

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.load_mode = LLAMA_LOAD_MODE_NONE;  // read weights into RAM, no mmap
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens(prompt.size() + 8);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(),
                                  tokens.data(), (int) tokens.size(),
                                  /*add_special*/ false, /*parse_special*/ true);
    if (n_tokens < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    tokens.resize(n_tokens);

    printf("{\"prompt_ids\": [");
    for (int i = 0; i < n_tokens; ++i) printf("%s%d", i ? ", " : "", tokens[i]);
    printf("], \"generated_ids\": [");

    std::vector<llama_token> generated;
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (int step = 0; step < n_predict; ++step) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) {
            if (logits[i] > logits[best]) best = i;
        }
        generated.push_back(best);
        printf("%s%d", step ? ", " : "", best);

        llama_token tok = best;
        batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    printf("]}\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
