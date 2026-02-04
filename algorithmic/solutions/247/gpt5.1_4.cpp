#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N;
    if (!(cin >> N)) return 0;

    vector<long long> A(N + 1), B(N + 1);
    for (int i = 1; i <= N; ++i) cin >> A[i];
    for (int i = 1; i <= N; ++i) cin >> B[i];

    long long sumA = 0, sumB = 0;
    for (int i = 1; i <= N; ++i) {
        sumA += A[i];
        sumB += B[i];
    }
    if (sumA != sumB) {
        cout << "No\n";
        return 0;
    }

    // N == 2 special case
    if (N == 2) {
        if (A[1] == B[1] && A[2] == B[2]) {
            cout << "Yes\n0\n";
            return 0;
        } else if (B[1] == A[2] - 1 && B[2] == A[1] + 1) {
            cout << "Yes\n1\n1 2\n";
            return 0;
        } else {
            cout << "No\n";
            return 0;
        }
    }

    // N >= 3
    vector<long long> cur = A;
    vector<pair<int,int>> ops;

    auto add_op = [&](int i, int j) {
        if (i > j) swap(i, j);
        long long xi = cur[i];
        long long xj = cur[j];
        cur[i] = xj - 1;
        cur[j] = xi + 1;
        ops.emplace_back(i, j);
    };

    auto move_j_to_1 = [&](int j) { // +1 to idx1, -1 from idx j
        if (j == 2) {
            // e1 - e2 using indices 1,2,3
            add_op(2, 3);
            add_op(1, 2);
            add_op(2, 3);
            add_op(1, 2);
            add_op(1, 3);
            add_op(2, 3);
        } else {
            // e1 - ej using indices 1,2,j  (j >= 3)
            add_op(1, 2);
            add_op(2, j);
            add_op(1, 2);
            add_op(1, j);
        }
    };

    auto move_1_to_j = [&](int j) { // -1 from idx1, +1 to idx j
        if (j == 2) {
            // e2 - e1 using indices 1,2,3 (inverse of above)
            add_op(2, 3);
            add_op(1, 3);
            add_op(1, 2);
            add_op(2, 3);
            add_op(1, 2);
            add_op(2, 3);
        } else {
            // ej - e1 using indices 1,2,j (j >= 3), inverse of previous macro
            add_op(1, j);
            add_op(1, 2);
            add_op(2, j);
            add_op(1, 2);
        }
    };

    vector<long long> diff(N + 1);
    for (int i = 1; i <= N; ++i) diff[i] = B[i] - A[i];

    for (int i = 2; i <= N; ++i) {
        if (diff[i] > 0) {
            for (int k = 0; k < diff[i]; ++k) {
                move_1_to_j(i);
            }
        } else if (diff[i] < 0) {
            for (int k = 0; k < -diff[i]; ++k) {
                move_j_to_1(i);
            }
        }
    }

    // At this point cur should equal B (guaranteed by construction)
    cout << "Yes\n";
    cout << ops.size() << '\n';
    for (auto &p : ops) {
        cout << p.first << ' ' << p.second << '\n';
    }

    return 0;
}