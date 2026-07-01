#pragma once
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

inline vector<vector<float>> load_fvecs(const string& path, int max_n = -1) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw runtime_error("No se pudo abrir: " + path);

    vector<vector<float>> vecs;
    int d = 0;

    while (true) {
        int dim = 0;
        if (fread(&dim, sizeof(int), 1, f) != 1) break;   
        if (d == 0) d = dim;
        else if (dim != d) break;

        vector<float> v(d);
        if (fread(v.data(), sizeof(float), d, f) != static_cast<size_t>(d)) break;
        vecs.push_back(move(v));

        if (max_n > 0 && static_cast<int>(vecs.size()) >= max_n) break;
    }
    fclose(f);
    return vecs;
}

inline vector<vector<int>> load_ivecs(const string& path, int max_n = -1) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw runtime_error("No se pudo abrir: " + path);

    vector<vector<int>> vecs;
    int d = 0;

    while (true) {
        int dim = 0;
        if (fread(&dim, sizeof(int), 1, f) != 1) break;
        if (d == 0) d = dim;

        vector<int> v(dim);
        if (fread(v.data(), sizeof(int), dim, f) != static_cast<size_t>(dim)) break;
        vecs.push_back(move(v));

        if (max_n > 0 && static_cast<int>(vecs.size()) >= max_n) break;
    }
    fclose(f);
    return vecs;
}