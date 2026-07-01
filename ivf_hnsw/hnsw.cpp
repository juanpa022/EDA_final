#include "hnsw.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace std;

HNSW::HNSW(const string& space, int M, int ef_construction, int ef)
    : space_(space), M_(M), M0_(2 * M), ef_construction_(ef_construction), ef_(ef),
      mL_(M > 1 ? 1.0 / log(static_cast<double>(M)) : 1.0), rng_(random_device{}()) {
    if (space_ != "l2" && space_ != "cosine" && space_ != "ip") throw invalid_argument("space debe ser 'l2', 'cosine' o 'ip'");
}

float HNSW::dist(const vector<float>& a, const vector<float>& b) const {
    if (space_ == "l2") {
        float s = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) { float d = a[i] - b[i]; s += d * d; }
        return s;
    }
if (space_ == "cosine") {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
    }
    return 1.0f - dot / (sqrt(na * nb) + 1e-10f);
}
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
return -dot;
}

void HNSW::normalize(vector<float>& v) const {
    float norm = 0.0f;
    for (float x : v) norm += x * x;
 norm = sqrt(norm) + 1e-10f;
    for (float& x : v) x /= norm;
}

int HNSW::random_layer() {
    double r = uniform_(rng_);
    if (r < 1e-10) r = 1e-10;
return static_cast<int>(-log(r) * mL_);
}

vector<HNSW::Pair> HNSW::search_layer(const vector<float>& query, const vector<int>& entry_points, int ef, int layer) const {
    if (layer >= static_cast<int>(graph_.size())) return {};
    const Layer& lg = graph_[layer]; unordered_set<int> visited;
    priority_queue<Pair, vector<Pair>, greater<Pair>> candidates;
    priority_queue<Pair> results;

    for (int ep : entry_points) {
if (data_.find(ep) == data_.end() || visited.count(ep)) continue;
        visited.insert(ep); float d = dist(query, data_.at(ep));
        candidates.push({d, ep}); results.push({d, ep});
    }

    while (!candidates.empty()) {
        auto [cd, cid] = candidates.top(); candidates.pop();
  if (!results.empty() && static_cast<int>(results.size()) >= ef) {
      if (cd > results.top().first) break;
  }
        auto it = lg.find(cid); if (it == lg.end()) continue;
        for (int nid : it->second) {
            if (visited.count(nid)) continue;
            visited.insert(nid); float d = dist(query, data_.at(nid));
            float worst = results.empty() ? numeric_limits<float>::max() : results.top().first;
  if (d < worst || static_cast<int>(results.size()) < ef) {
      candidates.push({d, nid}); results.push({d, nid});
      if (static_cast<int>(results.size()) > ef) results.pop();
  }
        }
    }
    vector<Pair> out; out.reserve(results.size());
    while (!results.empty()) { out.push_back(results.top()); results.pop(); }
    sort(out.begin(), out.end()); return out;
}

vector<int> HNSW::select_neighbors(const vector<float>& query, const vector<Pair>& candidates, int M) const {
    if (candidates.empty()) return {};
    vector<Pair> working = candidates; sort(working.begin(), working.end());
    vector<int> result; result.reserve(M);
    for (auto& [d_qe, e] : working) {
        if (static_cast<int>(result.size()) >= M) break;
        bool keep = true;
        for (int r : result) {
            if (dist(data_.at(e), data_.at(r)) < d_qe) { keep = false; break; }
        }
if (keep) result.push_back(e);
    }
    return result;
}

int HNSW::add_item(const vector<float>& embedding, int external_id) {
    vector<float> vec = embedding; if (space_ == "cosine") normalize(vec);
    int node_id = (external_id >= 0) ? external_id : next_id_;
    next_id_ = max(next_id_, node_id + 1); data_[node_id] = vec;
    int insert_layer = random_layer();
    while (static_cast<int>(graph_.size()) <= insert_layer) graph_.emplace_back();
    for (int l = 0; l <= insert_layer; ++l) graph_[l].emplace(node_id, vector<int>{});
    if (entry_point_ == -1) {
        entry_point_ = node_id; max_layer_ = insert_layer; return node_id;
    }
    vector<int> ep = {entry_point_};
    for (int layer = max_layer_; layer > insert_layer; --layer) {
        auto nearest = search_layer(vec, ep, 1, layer);
        if (!nearest.empty()) ep = {nearest[0].second};
    }
    for (int layer = min(insert_layer, max_layer_); layer >= 0; --layer) {
        int M_layer = (layer == 0) ? M0_ : M_;
        auto candidates = search_layer(vec, ep, ef_construction_, layer);
        auto neighbors = select_neighbors(vec, candidates, M_layer);
  graph_[layer][node_id] = neighbors;
        for (int nb : neighbors) {
            graph_[layer][nb].push_back(node_id);
            if (static_cast<int>(graph_[layer][nb].size()) > M_layer) {
                const auto& nb_emb = data_[nb]; vector<Pair> nb_cands; nb_cands.reserve(graph_[layer][nb].size());
                for (int c : graph_[layer][nb]) nb_cands.push_back({dist(nb_emb, data_[c]), c});
                graph_[layer][nb] = select_neighbors(nb_emb, nb_cands, M_layer);
            }
        }
        ep.clear(); ep.reserve(candidates.size());
        for (auto& [d, idx] : candidates) ep.push_back(idx);
    }
if (insert_layer > max_layer_) { max_layer_ = insert_layer; entry_point_ = node_id; }
    return node_id;
}

KNNResult HNSW::knn_query(const vector<float>& query, int k) const {
    vector<float> q = query; if (space_ == "cosine") normalize(q);
    if (entry_point_ == -1 || data_.empty()) return {};
    vector<int> ep = {entry_point_};
    for (int layer = max_layer_; layer > 0; --layer) {
        auto nearest = search_layer(q, ep, 1, layer);
  if (!nearest.empty()) ep = {nearest[0].second};
    }
    auto candidates = search_layer(q, ep, max(ef_, k), 0);
    if (static_cast<int>(candidates.size()) > k) candidates.resize(k);
    KNNResult res; res.ids.reserve(candidates.size()); res.distances.reserve(candidates.size());
    for (auto& [d, idx] : candidates) {
        res.ids.push_back(idx); res.distances.push_back(d);
    }
    return res;
}
