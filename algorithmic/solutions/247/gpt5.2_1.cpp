#include <bits/stdc++.h>
using namespace std;

struct Solver {
    int N;
    vector<long long> A, B;
    vector<pair<int,int>> ops; // 0-based indices stored as (i,j), i<j

    void applyOp(int i, int j) {
        // i<j
        long long x = A[i], y = A[j];
        A[i] = y - 1;
        A[j] = x + 1;
        ops.emplace_back(i, j);
    }

    vector<pair<int,int>> macroR2L(int i, int j) {
        // i<j, effect: A[i]+=1, A[j]-=1, others unchanged (requires N>=3)
        // Returns list of operations (each with first < second)
        vector<pair<int,int>> v;
        if (j - i >= 2) {
            int k = i + 1; // i < k < j
            v.emplace_back(i, k);
            v.emplace_back(k, j);
            v.emplace_back(i, k);
            v.emplace_back(i, j);
        } else {
            // adjacent: j == i+1
            if (i == 0) {
                int k = j + 1; // must exist since N>=3 and i==0 => j==1 => k==2
                v.emplace_back(i, j);
                v.emplace_back(i, k);
                v.emplace_back(j, k);
                v.emplace_back(i, k);
            } else {
                int k = i - 1; // exists
                v.emplace_back(i, j);
                v.emplace_back(k, j);
                v.emplace_back(k, i);
                v.emplace_back(k, j);
            }
        }
        return v;
    }

    void transfer1(int from, int to) {
        // Decrease A[from] by 1, increase A[to] by 1, keep others unchanged; N>=3 assumed
        if (from == to) return;
        if (to < from) {
            auto v = macroR2L(to, from); // from -> to (right to left)
            for (auto [i,j] : v) applyOp(i, j);
        } else {
            auto v = macroR2L(from, to); // this would do to -> from; invert it by reversing
            reverse(v.begin(), v.end());
            for (auto [i,j] : v) applyOp(i, j);
        }
    }

    void solve() {
        ios::sync_with_stdio(false);
        cin.tie(nullptr);

        cin >> N;
        A.assign(N, 0);
        B.assign(N, 0);
        for (int i = 0; i < N; i++) cin >> A[i];
        for (int i = 0; i < N; i++) cin >> B[i];

        long long sumA = 0, sumB = 0;
        for (int i = 0; i < N; i++) sumA += A[i], sumB += B[i];

        if (sumA != sumB) {
            cout << "No\n";
            return;
        }

        if (N == 2) {
            if (A == B) {
                cout << "Yes\n0\n";
                return;
            }
            vector<long long> T = A;
            // apply operation (1,2)
            long long x = T[0], y = T[1];
            T[0] = y - 1;
            T[1] = x + 1;
            if (T == B) {
                cout << "Yes\n1\n1 2\n";
            } else {
                cout << "No\n";
            }
            return;
        }

        // N >= 3: construct by unit transfers
        vector<int> surplus, deficit;
        surplus.reserve(N);
        deficit.reserve(N);
        for (int i = 0; i < N; i++) {
            if (A[i] > B[i]) surplus.push_back(i);
            else if (A[i] < B[i]) deficit.push_back(i);
        }

        size_t sp = 0, dp = 0;
        while (sp < surplus.size()) {
            int p = surplus[sp];
            int q = deficit[dp];
            long long canSend = min(A[p] - B[p], B[q] - A[q]);
            for (long long t = 0; t < canSend; t++) transfer1(p, q);
            if (A[p] == B[p]) sp++;
            if (A[q] == B[q]) dp++;
        }

        cout << "Yes\n";
        cout << ops.size() << "\n";
        for (auto [i, j] : ops) {
            cout << (i + 1) << " " << (j + 1) << "\n";
        }
    }
};

int main() {
    Solver s;
    s.solve();
    return 0;
}