#include "ivf_hnsw.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <omp.h>
#include <stdexcept>

using namespace std;

IVFHNSW::IVFHNSW(int d, int nlist, int nprobe, int M, int ef_construction, int ef,
                 const string& space)
    : d_(d),
      nlist_(nlist),
      nprobe_(min(nprobe, nlist)),
      centroid_hnsw_(space, M, ef_construction, nlist), 
      inverted_lists_(nlist) {
}

float IVFHNSW::vec_dist(const vector<float>& a, const vector<float>& b) const {
    float s = 0.0f;
    for (int i = 0; i < d_; ++i) {
        float diff = a[i] - b[i];
        s += diff * diff;
    }
    return s;
}

int IVFHNSW::nearest_centroid(const vector<float>& v) const {
    int best = 0;
    float best_d = numeric_limits<float>::max();
    for (int i = 0; i < nlist_; ++i) {
        float d = vec_dist(v, centroids_[i]);
        if (d < best_d) {
            best_d = d;
            best = i;
        }
    }
    return best;
}

vector<vector<float>> IVFHNSW::kmeans_pp(const vector<vector<float>>& X, int k,
                                         int n_iter) {
    int n = static_cast<int>(X.size());

    uniform_int_distribution<int> uni_int(0, n - 1);
    uniform_real_distribution<float> uni_real(0.0f, 1.0f);

    vector<vector<float>> centroids;
    centroids.reserve(k);
    centroids.push_back(X[uni_int(rng_)]);

    vector<float> min_dists(n, numeric_limits<float>::max());

    for (int c = 1; c < k; ++c) {
        const auto& last = centroids.back();
        float total = 0.0f;
        for (int i = 0; i < n; ++i) {
            float d = vec_dist(X[i], last);
            if (d < min_dists[i])
                min_dists[i] = d;
            total += min_dists[i];
        }
        float threshold = uni_real(rng_) * total;
        float accum = 0.0f;
        int chosen = n - 1;
        for (int i = 0; i < n; ++i) {
            accum += min_dists[i];
            if (accum >= threshold) {
                chosen = i;
                break;
            }
        }
        centroids.push_back(X[chosen]);
    }

    vector<int> assignments(n, 0);
    vector<int> counts(k, 0);

    for (int iter = 0; iter < n_iter; ++iter) {
        bool changed = false;
#pragma omp parallel for reduction(|| : changed) schedule(static)
        for (int i = 0; i < n; ++i) {
            float best_d = numeric_limits<float>::max();
            int best_c = 0;
            for (int c = 0; c < k; ++c) {
                float d = vec_dist(X[i], centroids[c]);
                if (d < best_d) {
                    best_d = d;
                    best_c = c;
                }
            }
            if (best_c != assignments[i]) {
                assignments[i] = best_c;
                changed = true;
            }
        }
        if (!changed)
            break;

        fill(counts.begin(), counts.end(), 0);
        for (auto& c : centroids) fill(c.begin(), c.end(), 0.0f);

#pragma omp parallel
        {
            vector<vector<float>> local_sum(k, vector<float>(d_, 0.0f));
            vector<int> local_cnt(k, 0);

#pragma omp for nowait schedule(static)
            for (int i = 0; i < n; ++i) {
                int c = assignments[i];
                ++local_cnt[c];
                for (int j = 0; j < d_; ++j) local_sum[c][j] += X[i][j];
            }

#pragma omp critical
            for (int c = 0; c < k; ++c) {
                counts[c] += local_cnt[c];
                for (int j = 0; j < d_; ++j) centroids[c][j] += local_sum[c][j];
            }
        }

        for (int c = 0; c < k; ++c) {
            if (counts[c] == 0) {
                centroids[c] = X[uni_int(rng_)]; 
            } else {
                float inv = 1.0f / static_cast<float>(counts[c]);
                for (float& v : centroids[c]) v *= inv;
            }
        }
    }
    return centroids;
}

void IVFHNSW::train(const vector<vector<float>>& X, int n_iter) {
    if (X.empty())
        throw invalid_argument("train: X vacío");
    if (static_cast<int>(X[0].size()) != d_)
        throw invalid_argument("train: dimensión incorrecta");
    if (static_cast<int>(X.size()) < nlist_)
        throw invalid_argument("train: menos muestras que clusters");

    centroids_ = kmeans_pp(X, nlist_, n_iter);

    for (int i = 0; i < nlist_; ++i) centroid_hnsw_.add_item(centroids_[i], i);

    trained_ = true;
}

void IVFHNSW::add(const vector<vector<float>>& X, const vector<int>& ids) {
    if (!trained_)
        throw runtime_error("add: llamar train() primero");

    for (int i = 0; i < static_cast<int>(X.size()); ++i) {
        int ext_id = ids.empty() ? next_id_++ : ids[i];
        int cluster = nearest_centroid(X[i]);
        inverted_lists_[cluster].push_back({ext_id, X[i]});
    }
    if (!ids.empty())
        next_id_ = ids.back() + 1;
}

KNNResult IVFHNSW::search(const vector<float>& query, int k) const {
    if (!trained_)
        throw runtime_error("search: llamar train() primero");

    KNNResult centroid_res = centroid_hnsw_.knn_query(query, nprobe_);

    vector<pair<float, int>> candidates;
    for (int cid : centroid_res.ids) {
        for (const auto& [ext_id, vec] : inverted_lists_[cid]) {
            float d = vec_dist(query, vec);
            candidates.push_back({d, ext_id});
        }
    }

    if (static_cast<int>(candidates.size()) > k) {
        partial_sort(candidates.begin(), candidates.begin() + k, candidates.end());
        candidates.resize(k);
    } else {
        sort(candidates.begin(), candidates.end());
    }

    KNNResult res;
    res.ids.reserve(candidates.size());
    res.distances.reserve(candidates.size());
    for (auto& [d, id] : candidates) {
        res.ids.push_back(id);
        res.distances.push_back(d);
    }
    return res;
}

int IVFHNSW::ntotal() const {
    int total = 0;
    for (const auto& lst : inverted_lists_) total += static_cast<int>(lst.size());
    return total;
}