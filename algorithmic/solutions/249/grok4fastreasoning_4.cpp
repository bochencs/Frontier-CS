#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin >> n;
    vector<int> p(n + 1);
    vector<int> o1(n + 1, 0);
    for (int j = 2; j <= n; j++) {
        cout << "? 1 " << j << endl;
        fflush(stdout);
        int oo;
        cin >> oo;
        if (oo == -1) return 0;
        o1[j] = oo;
    }
    p[1] = o1[2];
    for (int j = 3; j <= n; j++) {
        p[1] &= o1[j];
    }
    int min_added = INT_MAX;
    int second_hub = 2;
    for (int j = 2; j <= n; j++) {
        int added = o1[j] & ~p[1];
        if (added < min_added) {
            min_added = added;
            second_hub = j;
        }
    }
    vector<int> o2(n + 1, 0);
    for (int j = 1; j <= n; j++) {
        if (j == second_hub) continue;
        cout << "? " << second_hub << " " << j << endl;
        fflush(stdout);
        int oo;
        cin >> oo;
        if (oo == -1) return 0;
        o2[j] = oo;
    }
    p[second_hub] = o2[1];
    for (int j = 1; j <= n; j++) {
        if (j == second_hub) continue;
        p[second_hub] &= o2[j];
    }
    int m = p[1] & p[second_hub];
    vector<int> bits;
    for (int bb = 0; bb < 11; bb++) {
        if (m & (1 << bb)) bits.push_back(bb);
    }
    int kk = bits.size();
    set<int> used_p;
    used_p.insert(p[1]);
    used_p.insert(p[second_hub]);
    vector<int> base(n + 1, 0);
    for (int j = 1; j <= n; j++) {
        if (j == 1 || j == second_hub) continue;
        base[j] = (o1[j] & ~p[1]) | (o2[j] & ~p[second_hub]);
    }
    for (int j = 1; j <= n; j++) {
        if (j == 1 || j == second_hub) continue;
        vector<int> possible;
        for (int t = 0; t < (1 << kk); t++) {
            int mask = 0;
            for (int ii = 0; ii < kk; ii++) {
                if (t & (1 << ii)) mask |= (1 << bits[ii]);
            }
            int x = base[j] | mask;
            if (x < n && used_p.find(x) == used_p.end()) {
                possible.push_back(x);
            }
        }
        // Assume exactly one
        sort(possible.begin(), possible.end());
        p[j] = possible[0];
        used_p.insert(p[j]);
    }
    cout << "!";
    for (int j = 1; j <= n; j++) {
        cout << " " << p[j];
    }
    cout << endl;
    fflush(stdout);
    return 0;
}