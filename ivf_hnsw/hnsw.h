#pragma once
#include <cmath>
#include <limits>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

struct KNNResult { vector<int> ids; vector<float> distances; };

class HNSW {
public:

    explicit HNSW(const string& space = "l2", int M = 16, int ef_construction = 200, int ef = 50);
    int add_item(const vector<float>& embedding, int external_id = -1);
KNNResult knn_query(const vector<float>& query, int k) const;
    int size() const { return static_cast<int>(data_.size()); }
    bool empty() const { return data_.empty(); }
  int max_layer() const { return max_layer_; }

private:

    using Layer = unordered_map<int, vector<int>>;
using Pair = pair<float, int>;

    string space_;
    int M_, M0_, ef_construction_, ef_;
    double mL_;
  vector<Layer> graph_; unordered_map<int, vector<float>> data_;
    int entry_point_ = -1;
    int max_layer_ = -1;
int next_id_ = 0;

    mutable mt19937 rng_;
    uniform_real_distribution<double> uniform_{0.0, 1.0};
    float dist(const vector<float>& a, const vector<float>& b) const;
void normalize(vector<float>& v) const;

    int random_layer();
    vector<Pair> search_layer(const vector<float>& query, const vector<int>& entry_points, int ef, int layer) const;
    vector<int> select_neighbors(const vector<float>& query, const vector<Pair>& candidates, int M) const;
};