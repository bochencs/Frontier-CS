#include <bits/stdc++.h>
using namespace std;

static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

enum Act { LEFT = 0, RIGHT = 1, STEP = 2 };

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int r, c;
    if (!(cin >> r >> c)) return 0;
    vector<string> g(r);
    for (int i = 0; i < r; i++) cin >> g[i];

    vector<vector<int>> cellId(r, vector<int>(c, -1));
    vector<int> cellRow, cellCol;
    cellRow.reserve(r * c);
    cellCol.reserve(r * c);

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            if (g[i][j] == '.') {
                int id = (int)cellRow.size();
                cellId[i][j] = id;
                cellRow.push_back(i);
                cellCol.push_back(j);
            }
        }
    }

    int M = (int)cellRow.size();
    if (M == 0) return 0;

    vector<array<int,4>> stateOfCell(M);
    int N = M * 4;
    for (int cell = 0; cell < M; cell++) {
        for (int d = 0; d < 4; d++) stateOfCell[cell][d] = cell * 4 + d;
    }

    static const int dx[4] = {-1, 0, 1, 0};
    static const int dy[4] = {0, 1, 0, -1};

    vector<int> distWall(N, 0);
    vector<int> leftT(N), rightT(N), stepT(N);
    vector<int> cellOfState(N), dirOfState(N);

    auto isWall = [&](int x, int y) -> bool {
        if (x < 0 || x >= r || y < 0 || y >= c) return true;
        return g[x][y] == '#';
    };

    for (int cell = 0; cell < M; cell++) {
        int x0 = cellRow[cell], y0 = cellCol[cell];
        for (int d = 0; d < 4; d++) {
            int s = stateOfCell[cell][d];
            cellOfState[s] = cell;
            dirOfState[s] = d;

            leftT[s] = stateOfCell[cell][(d + 3) & 3];
            rightT[s] = stateOfCell[cell][(d + 1) & 3];

            int nx = x0 + dx[d], ny = y0 + dy[d];
            if (!isWall(nx, ny)) {
                int ncell = cellId[nx][ny];
                stepT[s] = stateOfCell[ncell][d];
            } else {
                stepT[s] = -1;
            }

            int cnt = 0;
            int x = x0, y = y0;
            while (true) {
                int tx = x + dx[d], ty = y + dy[d];
                if (isWall(tx, ty)) break;
                cnt++;
                x = tx; y = ty;
                if (cnt >= 99) break;
            }
            distWall[s] = cnt;
        }
    }

    // Compute observational equivalence classes (Moore machine minimization with step disabled when dist==0).
    vector<int> cls(N, 0), newcls(N, 0);
    vector<int> distToClass(100, -1);
    int initClasses = 0;
    for (int s = 0; s < N; s++) {
        int d = distWall[s];
        if (distToClass[d] == -1) distToClass[d] = initClasses++;
        cls[s] = distToClass[d];
    }

    while (true) {
        unordered_map<uint64_t, int> mp;
        mp.reserve((size_t)N * 2);

        int nextId = 0;
        bool same = true;
        for (int s = 0; s < N; s++) {
            int out = distWall[s];
            int l = cls[leftT[s]];
            int rcls = cls[rightT[s]];
            int st = (distWall[s] > 0 ? cls[stepT[s]] : -1);

            uint64_t key = ((uint64_t)out << 48) |
                           ((uint64_t)l << 32) |
                           ((uint64_t)rcls << 16) |
                           (uint64_t)(st + 1);

            auto it = mp.find(key);
            int id;
            if (it == mp.end()) {
                id = nextId++;
                mp.emplace(key, id);
            } else {
                id = it->second;
            }
            newcls[s] = id;
            if (newcls[s] != cls[s]) same = false;
        }
        cls.swap(newcls);
        if (same) break;
    }

    int numClasses = 0;
    for (int s = 0; s < N; s++) numClasses = max(numClasses, cls[s] + 1);

    vector<int> cellMark(M, 0);
    int cellMarkTag = 1;

    vector<int> classStamp(numClasses, 0), classCell(numClasses, -1);
    int classTag = 1;

    vector<int> tmpCellStamp(M, 0);
    int tmpCellTag = 1;

    unordered_map<uint64_t, int> seenBeliefs;
    seenBeliefs.reserve(1 << 16);

    vector<int> belief(N);
    iota(belief.begin(), belief.end(), 0);

    auto beliefHash = [&](const vector<int>& B, int d) -> uint64_t {
        uint64_t h = splitmix64((uint64_t)B.size() ^ (uint64_t)(d + 1234567));
        for (int s : B) h ^= splitmix64((uint64_t)s + 0x9e3779b97f4a7c15ULL);
        return h;
    };

    auto actStr = [&](int act) -> const char* {
        if (act == LEFT) return "left";
        if (act == RIGHT) return "right";
        return "step";
    };

    auto trans = [&](int s, int act) -> int {
        if (act == LEFT) return leftT[s];
        if (act == RIGHT) return rightT[s];
        return stepT[s];
    };

    auto chooseAction = [&](const vector<int>& B, int curd, uint64_t bh) -> int {
        struct Cand {
            int act;
            int worstCells;
            int worstStates;
            int groups;
            int pref;
        };

        array<vector<int>, 100> buckets;
        array<int, 100> usedFlags{};
        vector<int> used;
        used.reserve(100);

        auto evalAct = [&](int act) -> Cand {
            used.clear();
            // We'll lazily clear buckets only when first used per dist.
            for (int dd : used) { (void)dd; } // no-op; used cleared.

            int worstCells = 0, worstStates = 0, groups = 0;

            for (int s : B) {
                int ns = trans(s, act);
                int dd = distWall[ns];
                if (!usedFlags[dd]) {
                    usedFlags[dd] = 1;
                    used.push_back(dd);
                    buckets[dd].clear();
                }
                buckets[dd].push_back(cellOfState[ns]);
            }

            for (int dd : used) {
                int sz = (int)buckets[dd].size();
                worstStates = max(worstStates, sz);
                groups++;

                int tag = ++tmpCellTag;
                if (tmpCellTag == INT_MAX) {
                    tmpCellTag = 1;
                    fill(tmpCellStamp.begin(), tmpCellStamp.end(), 0);
                    tag = ++tmpCellTag;
                }

                int uniq = 0;
                for (int cell : buckets[dd]) {
                    if (tmpCellStamp[cell] != tag) {
                        tmpCellStamp[cell] = tag;
                        uniq++;
                    }
                }
                worstCells = max(worstCells, uniq);
            }

            for (int dd : used) usedFlags[dd] = 0;

            int pref = 0;
            if (act == STEP) pref = 0;
            else if (act == RIGHT) pref = 1;
            else pref = 2;

            return Cand{act, worstCells, worstStates, groups, pref};
        };

        vector<int> acts;
        acts.push_back(LEFT);
        acts.push_back(RIGHT);
        if (curd > 0) acts.push_back(STEP);

        vector<Cand> cands;
        cands.reserve(3);
        for (int act : acts) cands.push_back(evalAct(act));

        sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
            if (a.worstCells != b.worstCells) return a.worstCells < b.worstCells;
            if (a.worstStates != b.worstStates) return a.worstStates < b.worstStates;
            if (a.groups != b.groups) return a.groups > b.groups;
            return a.pref < b.pref;
        });

        int cnt = ++seenBeliefs[bh];
        if ((cnt >= 6) && (int)cands.size() >= 2) {
            // Try to escape cycles by occasionally taking the 2nd best action.
            if (cnt % 6 == 0) return cands[1].act;
        }
        return cands[0].act;
    };

    int d;
    while (cin >> d) {
        if (d == -1) return 0;

        // Filter belief by observed distance.
        vector<int> nb;
        nb.reserve(belief.size());
        for (int s : belief) if (distWall[s] == d) nb.push_back(s);
        belief.swap(nb);

        if (belief.empty()) {
            cout << "no" << endl;
            return 0;
        }

        // Unique cell check.
        int tag = ++cellMarkTag;
        if (cellMarkTag == INT_MAX) {
            cellMarkTag = 1;
            fill(cellMark.begin(), cellMark.end(), 0);
            tag = ++cellMarkTag;
        }
        int uniqueCell = -1, cellCnt = 0;
        for (int s : belief) {
            int cell = cellOfState[s];
            if (cellMark[cell] != tag) {
                cellMark[cell] = tag;
                uniqueCell = cell;
                cellCnt++;
                if (cellCnt > 1) break;
            }
        }
        if (cellCnt == 1) {
            cout << "yes " << (cellRow[uniqueCell] + 1) << " " << (cellCol[uniqueCell] + 1) << endl;
            return 0;
        }

        // If two different cells are still possible but belong to the same equivalence class, impossible to resolve.
        int ctag = ++classTag;
        if (classTag == INT_MAX) {
            classTag = 1;
            fill(classStamp.begin(), classStamp.end(), 0);
            ctag = ++classTag;
        }
        bool impossibleNow = false;
        for (int s : belief) {
            int k = cls[s];
            int cell = cellOfState[s];
            if (classStamp[k] != ctag) {
                classStamp[k] = ctag;
                classCell[k] = cell;
            } else {
                if (classCell[k] != cell) {
                    impossibleNow = true;
                    break;
                }
            }
        }
        if (impossibleNow) {
            cout << "no" << endl;
            return 0;
        }

        uint64_t bh = beliefHash(belief, d);
        int act = chooseAction(belief, d, bh);

        cout << actStr(act) << endl;

        // Apply action to belief (without next observation filtering yet).
        vector<int> after;
        after.reserve(belief.size());
        if (act == STEP) {
            // Must be safe: only step when d>0
            if (d == 0) {
                cout << "no" << endl;
                return 0;
            }
            for (int s : belief) {
                int ns = stepT[s];
                if (ns < 0) { // should not happen if filtered correctly
                    cout << "no" << endl;
                    return 0;
                }
                after.push_back(ns);
            }
        } else if (act == LEFT) {
            for (int s : belief) after.push_back(leftT[s]);
        } else {
            for (int s : belief) after.push_back(rightT[s]);
        }
        belief.swap(after);
    }

    return 0;
}