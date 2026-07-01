#include "ivf_hnsw/hnsw.h"
#include "ivf_hnsw/ivf_hnsw.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <random>
#include <unordered_set>
#include <vector>

using namespace std;

struct Timer {
    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
    double ms() const {
        return chrono::duration<double, milli>(
            chrono::steady_clock::now() - t0
        ).count();
    }
    void reset() { t0 = chrono::steady_clock::now(); }
};

vector<vector<float>> random_vectors(int n, int d, mt19937& rng) {
    normal_distribution<float> nd;
    vector<vector<float>> X(n, vector<float>(d));
    for (auto& v : X)
        for (float& x : v) x = nd(rng);
    return X;
}

vector<vector<int>> brute_force_knn(
    const vector<vector<float>>& data,
    const vector<vector<float>>& queries,
    int k
) {
    int nq = static_cast<int>(queries.size());
    int nd = static_cast<int>(data.size());
    int d  = static_cast<int>(data[0].size());
    vector<vector<int>> gt(nq);

    #pragma omp parallel for schedule(dynamic)
    for (int qi = 0; qi < nq; ++qi) {
        const auto& q = queries[qi];
        vector<pair<float, int>> dists;
        dists.reserve(nd);
        for (int i = 0; i < nd; ++i) {
            float s = 0;
            for (int j = 0; j < d; ++j) {
                float diff = q[j] - data[i][j];
                s += diff * diff;
            }
            dists.push_back({s, i});
        }
        partial_sort(dists.begin(), dists.begin() + k, dists.end());
        vector<int> ids(k);
        for (int i = 0; i < k; ++i) ids[i] = dists[i].second;
        gt[qi] = move(ids);
    }
    return gt;
}

float recall_at_k(
    const vector<vector<int>>& gt,
    const vector<KNNResult>&   preds,
    int k
) {
    float total = 0.0f;
    int   n     = static_cast<int>(gt.size());
    for (int q = 0; q < n; ++q) {
        unordered_set<int> gt_set(gt[q].begin(), gt[q].begin() + k);
        int hits = 0;
        for (int i = 0; i < k && i < static_cast<int>(preds[q].ids.size()); ++i)
            if (gt_set.count(preds[q].ids[i])) ++hits;
        total += static_cast<float>(hits) / k;
    }
    return total / n;
}

int main() {
    mt19937 rng(42);

    constexpr int D       = 100;    
    constexpr int N_DATA  = 10000;  
    constexpr int N_QUERY = 200;    
    constexpr int K       = 10;     
    constexpr int NLIST   = 50;     
    constexpr int NPROBE  = 8;      

    cout << "=== IVF+HNSW demo ===\n"
              << "dim=" << D << "  n_data=" << N_DATA
              << "  n_queries=" << N_QUERY << "  k=" << K << "\n\n";

    auto data    = random_vectors(N_DATA,  D, rng);
    auto queries = random_vectors(N_QUERY, D, rng);

    cout << "--- Brute-Force (ground truth) ---\n";
    Timer t;
    auto gt = brute_force_knn(data, queries, K);
    cout << "  query: " << fixed << setprecision(1)
              << t.ms() << " ms\n\n";

    cout << "--- HNSW  (M=16, ef_construction=200, ef=50) ---\n";
    HNSW hnsw("l2", 16, 200, 50);
    t.reset();
    for (int i = 0; i < N_DATA; ++i) hnsw.add_item(data[i], i);
    cout << "  build : " << t.ms() << " ms\n";

    vector<KNNResult> hnsw_res(N_QUERY);
    t.reset();
    #pragma omp parallel for schedule(dynamic)
    for (int qi = 0; qi < N_QUERY; ++qi)
        hnsw_res[qi] = hnsw.knn_query(queries[qi], K);
    cout << "  query : " << t.ms() << " ms"
              << "  |  Recall@" << K << " = "
              << setprecision(3) << recall_at_k(gt, hnsw_res, K) << "\n\n";

    cout << "--- IVF+HNSW  (nlist=" << NLIST
              << ", nprobe=" << NPROBE
              << ", M=16, ef_construction=200) ---\n";
    IVFHNSW ivf(D, NLIST, NPROBE, 16, 200, 50, "l2");

    vector<vector<float>> train_set(data.begin(),
                                      data.begin() + N_DATA / 2);
    t.reset();
    ivf.train(train_set, 25);
    cout << "  train : " << t.ms() << " ms\n";

    t.reset();
    ivf.add(data);
    cout << "  add   : " << t.ms() << " ms"
              << "  (" << ivf.ntotal() << " vectores)\n";

    vector<KNNResult> ivf_res(N_QUERY);
    t.reset();
    #pragma omp parallel for schedule(dynamic)
    for (int qi = 0; qi < N_QUERY; ++qi)
        ivf_res[qi] = ivf.search(queries[qi], K);
    cout << "  query : " << t.ms() << " ms"
              << "  |  Recall@" << K << " = "
              << setprecision(3) << recall_at_k(gt, ivf_res, K) << "\n";

    return 0;
}