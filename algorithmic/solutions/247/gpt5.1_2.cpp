#include <bits/stdc++.h>
using namespace std;

// For N == 3 BFS
const int SHIFT = 600;
const int MAXC = 2 * SHIFT + 1;

static bool vis3[MAXC][MAXC];
static short previ3[MAXC][MAXC];
static short prevj3[MAXC][MAXC];
static signed char opi3[MAXC][MAXC];
static signed char opj3[MAXC][MAXC];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N;
    if (!(cin >> N)) return 0;
    vector<long long> A(N + 1), B(N + 1);
    long long sumA = 0, sumB = 0;
    for (int i = 1; i <= N; ++i) {
        cin >> A[i];
        sumA += A[i];
    }
    for (int i = 1; i <= N; ++i) {
        cin >> B[i];
        sumB += B[i];
    }

    if (sumA != sumB) {
        cout << "No\n";
        return 0;
    }

    if (N == 2) {
        if (A[1] == B[1] && A[2] == B[2]) {
            cout << "Yes\n0\n";
            return 0;
        }
        long long x1 = A[1], x2 = A[2];
        long long nx1 = x2 - 1, nx2 = x1 + 1;
        if (nx1 == B[1] && nx2 == B[2]) {
            cout << "Yes\n1\n1 2\n";
            return 0;
        } else {
            cout << "No\n";
            return 0;
        }
    }

    if (N == 3) {
        long long S = sumA; // == sumB
        // Initialize BFS arrays
        for (int i = 0; i < MAXC; ++i) {
            memset(vis3[i], 0, MAXC * sizeof(bool));
        }

        auto isValid = [&](int x1, int x2) -> bool {
            long long x3 = S - x1 - x2;
            if (x1 < -SHIFT || x1 > SHIFT) return false;
            if (x2 < -SHIFT || x2 > SHIFT) return false;
            if (x3 < -SHIFT || x3 > SHIFT) return false;
            return true;
        };

        int startX = (int)A[1], startY = (int)A[2];
        int goalX = (int)B[1], goalY = (int)B[2];

        int sI = startX + SHIFT;
        int sJ = startY + SHIFT;
        int gI = goalX + SHIFT;
        int gJ = goalY + SHIFT;

        queue<pair<int,int>> q;
        vis3[sI][sJ] = true;
        previ3[sI][sJ] = -1;
        prevj3[sI][sJ] = -1;
        opi3[sI][sJ] = opj3[sI][sJ] = 0;
        q.emplace(sI, sJ);

        auto tryPush = [&](int nx1, int nx2, int fromI, int fromJ, int oi, int oj) {
            if (!isValid(nx1, nx2)) return;
            int nI = nx1 + SHIFT;
            int nJ = nx2 + SHIFT;
            if (nI < 0 || nI >= MAXC || nJ < 0 || nJ >= MAXC) return;
            if (vis3[nI][nJ]) return;
            vis3[nI][nJ] = true;
            previ3[nI][nJ] = (short)fromI;
            prevj3[nI][nJ] = (short)fromJ;
            opi3[nI][nJ] = (signed char)oi;
            opj3[nI][nJ] = (signed char)oj;
            q.emplace(nI, nJ);
        };

        while (!q.empty() && !vis3[gI][gJ]) {
            auto [I, J] = q.front(); q.pop();
            int x1 = I - SHIFT;
            int x2 = J - SHIFT;
            long long x3 = S - x1 - x2;

            // operation on (1,2)
            {
                int nx1 = x2 - 1;
                int nx2 = x1 + 1;
                tryPush(nx1, nx2, I, J, 1, 2);
            }
            // operation on (1,3)
            {
                int nx1 = (int)(x3 - 1);
                int nx2 = x2;
                tryPush(nx1, nx2, I, J, 1, 3);
            }
            // operation on (2,3)
            {
                int nx1 = x1;
                int nx2 = (int)(x3 - 1);
                tryPush(nx1, nx2, I, J, 2, 3);
            }
        }

        if (!vis3[gI][gJ]) {
            // Should be reachable in theory, but domain might be too small.
            cout << "No\n";
            return 0;
        }

        vector<pair<int,int>> ops;
        int ci = gI, cj = gJ;
        while (!(ci == sI && cj == sJ)) {
            int oi = opi3[ci][cj];
            int oj = opj3[ci][cj];
            ops.emplace_back(oi, oj);
            int pi = previ3[ci][cj];
            int pj = prevj3[ci][cj];
            ci = pi; cj = pj;
        }
        reverse(ops.begin(), ops.end());

        cout << "Yes\n";
        cout << (int)ops.size() << "\n";
        for (auto &p : ops) {
            cout << p.first << ' ' << p.second << "\n";
        }
        return 0;
    }

    // N >= 4 : use algebraic translation construction
    vector<long long> D(N + 1);
    for (int i = 1; i <= N; ++i) D[i] = B[i] - A[i];

    vector<long long> alpha(N + 1, 0);
    for (int k = 3; k <= N - 1; ++k) {
        alpha[k] = -D[k];
    }
    long long beta = D[2];
    alpha[N] = -D[N] - beta;

    vector<pair<int,int>> ops;
    auto add_T = [&](int x, int y) {
        if (x > y) swap(x, y);
        ops.emplace_back(x, y);
    };

    auto do_v = [&](int k, int sign) {
        // v_k = e1 - e_k, using indices (1,2,k)
        if (sign == 1) {
            // from k to 1
            add_T(1, 2);
            add_T(2, k);
            add_T(1, 2);
            add_T(1, k);
        } else {
            // from 1 to k (inverse)
            add_T(1, k);
            add_T(1, 2);
            add_T(2, k);
            add_T(1, 2);
        }
    };

    auto do_w = [&](int sign) {
        int kN = N;
        // w = e2 - eN, using indices (2,3,N)
        if (sign == 1) {
            // from N to 2
            add_T(2, 3);
            add_T(3, kN);
            add_T(2, 3);
            add_T(2, kN);
        } else {
            // from 2 to N
            add_T(2, kN);
            add_T(2, 3);
            add_T(3, kN);
            add_T(2, 3);
        }
    };

    for (int k = 3; k <= N; ++k) {
        long long c = alpha[k];
        if (c > 0) {
            for (long long t = 0; t < c; ++t) do_v(k, +1);
        } else if (c < 0) {
            for (long long t = 0; t < -c; ++t) do_v(k, -1);
        }
    }
    if (beta > 0) {
        for (long long t = 0; t < beta; ++t) do_w(+1);
    } else if (beta < 0) {
        for (long long t = 0; t < -beta; ++t) do_w(-1);
    }

    cout << "Yes\n";
    cout << (int)ops.size() << "\n";
    for (auto &p : ops) {
        cout << p.first << ' ' << p.second << "\n";
    }
    return 0;
}