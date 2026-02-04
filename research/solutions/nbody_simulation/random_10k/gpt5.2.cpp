#include "world.h"
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

class MySimulator : public Simulator {
private:
    int numThreads = 16;
    int N = 0;

    std::vector<float> posX, posY, mass;

    float cellSize = 1.0f;
    float invCellSize = 1.0f;
    float minX = 0.0f, minY = 0.0f;
    int nx = 0, ny = 0, numCells = 0;
    int range = 1;

    std::vector<int> particleCell;
    std::vector<int> cellCounts;
    std::vector<int> cellOffsets; // size numCells+1
    std::vector<int> cursor;      // size numCells
    std::vector<int> sortedIndices;

    int prevNx = -1, prevNy = -1, prevRange = -1;
    int maxNeighbors = 0;
    std::vector<int> neighCount; // size numCells
    std::vector<int> neighList;  // size numCells*maxNeighbors

    inline void rebuildNeighborListIfNeeded() {
        if (nx == prevNx && ny == prevNy && range == prevRange && (int)neighCount.size() == numCells) return;

        prevNx = nx;
        prevNy = ny;
        prevRange = range;

        maxNeighbors = (2 * range + 1) * (2 * range + 1);
        neighCount.assign(numCells, 0);
        neighList.assign((size_t)numCells * (size_t)maxNeighbors, 0);

        for (int cy = 0; cy < ny; ++cy) {
            for (int cx = 0; cx < nx; ++cx) {
                int cell = cy * nx + cx;
                int k = 0;
                for (int dy = -range; dy <= range; ++dy) {
                    int y = cy + dy;
                    if ((unsigned)y >= (unsigned)ny) continue;
                    int row = y * nx;
                    for (int dx = -range; dx <= range; ++dx) {
                        int x = cx + dx;
                        if ((unsigned)x >= (unsigned)nx) continue;
                        neighList[(size_t)cell * (size_t)maxNeighbors + (size_t)k] = row + x;
                        ++k;
                    }
                }
                neighCount[cell] = k;
            }
        }
    }

    inline void buildGrid(const StepParameters params) {
        float mx = std::numeric_limits<float>::infinity();
        float my = std::numeric_limits<float>::infinity();
        float Mx = -std::numeric_limits<float>::infinity();
        float My = -std::numeric_limits<float>::infinity();

        for (int i = 0; i < N; ++i) {
            float x = posX[i];
            float y = posY[i];
            if (x < mx) mx = x;
            if (y < my) my = y;
            if (x > Mx) Mx = x;
            if (y > My) My = y;
        }

        float cull = params.cullRadius;
        float cs = std::max(cull * 0.5f, 1e-3f);
        cs = std::min(cs, std::max(cull, 1e-3f));

        const float pad = 1e-3f;
        minX = mx - pad;
        minY = my - pad;
        float maxX = Mx + pad;
        float maxY = My + pad;

        float width = std::max(maxX - minX, 1e-3f);
        float height = std::max(maxY - minY, 1e-3f);

        auto computeDims = [&](float cellSz) {
            int tx = (int)(width / cellSz) + 1;
            int ty = (int)(height / cellSz) + 1;
            if (tx < 1) tx = 1;
            if (ty < 1) ty = 1;
            return std::pair<int,int>(tx, ty);
        };

        auto [tx, ty] = computeDims(cs);
        size_t cells = (size_t)tx * (size_t)ty;

        const size_t cellLimit = 65536;
        if (cells > cellLimit) {
            float scale = std::sqrt((float)cells / (float)cellLimit);
            cs *= scale;
            auto dims2 = computeDims(cs);
            tx = dims2.first;
            ty = dims2.second;
            cells = (size_t)tx * (size_t)ty;
        }

        nx = tx;
        ny = ty;
        numCells = (int)cells;

        cellSize = cs;
        invCellSize = 1.0f / cellSize;
        range = (int)std::ceil(cull / cellSize);
        if (range < 1) range = 1;

        particleCell.resize((size_t)N);
        sortedIndices.resize((size_t)N);

        if ((int)cellCounts.size() != numCells) cellCounts.assign((size_t)numCells, 0);
        else std::fill(cellCounts.begin(), cellCounts.end(), 0);

        if ((int)cellOffsets.size() != numCells + 1) cellOffsets.assign((size_t)numCells + 1, 0);
        else std::fill(cellOffsets.begin(), cellOffsets.end(), 0);

        for (int i = 0; i < N; ++i) {
            int cx = (int)((posX[i] - minX) * invCellSize);
            int cy = (int)((posY[i] - minY) * invCellSize);
            if (cx < 0) cx = 0; else if (cx >= nx) cx = nx - 1;
            if (cy < 0) cy = 0; else if (cy >= ny) cy = ny - 1;
            int c = cy * nx + cx;
            particleCell[i] = c;
            cellCounts[c] += 1;
        }

        int sum = 0;
        for (int c = 0; c < numCells; ++c) {
            cellOffsets[c] = sum;
            sum += cellCounts[c];
        }
        cellOffsets[numCells] = sum;

        if ((int)cursor.size() != numCells) cursor.resize((size_t)numCells);
        std::copy(cellOffsets.begin(), cellOffsets.begin() + numCells, cursor.begin());

        for (int i = 0; i < N; ++i) {
            int c = particleCell[i];
            int dst = cursor[c]++;
            sortedIndices[(size_t)dst] = i;
        }

        rebuildNeighborListIfNeeded();
    }

public:
    void init(int numParticles, StepParameters params) override {
        (void)params;
        N = numParticles;

        int procs = omp_get_num_procs();
        numThreads = std::min(16, std::max(1, procs));
        omp_set_dynamic(0);
        omp_set_num_threads(numThreads);

        posX.resize((size_t)N);
        posY.resize((size_t)N);
        mass.resize((size_t)N);

        particleCell.resize((size_t)N);
        sortedIndices.resize((size_t)N);

        cellCounts.clear();
        cellOffsets.clear();
        cursor.clear();
        neighCount.clear();
        neighList.clear();

        prevNx = prevNy = prevRange = -1;
        maxNeighbors = 0;
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        const int n = (int)particles.size();
        if (n != N) {
            init(n, params);
        }
        if ((int)newParticles.size() != N) newParticles.resize((size_t)N);

        Particle* __restrict p = particles.data();
        Particle* __restrict out = newParticles.data();

        float* __restrict px = posX.data();
        float* __restrict py = posY.data();
        float* __restrict pm = mass.data();

        #pragma omp parallel for schedule(static, 256)
        for (int i = 0; i < N; ++i) {
            px[i] = p[i].position.x;
            py[i] = p[i].position.y;
            pm[i] = p[i].mass;
        }

        buildGrid(params);

        const float dt = params.deltaTime;
        const float cull = params.cullRadius;
        const float cull2 = cull * cull;
        const float cull075 = cull * 0.75f;
        const float invC25 = (cull > 0.0f) ? (1.0f / (cull * 0.25f)) : 0.0f;

        const int* __restrict pCell = particleCell.data();
        const int* __restrict offsets = cellOffsets.data();
        const int* __restrict sidx = sortedIndices.data();
        const int* __restrict nCount = neighCount.data();
        const int* __restrict nList = neighList.data();
        const int maxN = maxNeighbors;

        #pragma omp parallel for schedule(static, 32)
        for (int i = 0; i < N; ++i) {
            const float pix = px[i];
            const float piy = py[i];
            const float mi = pm[i];

            float fx = 0.0f, fy = 0.0f;

            const int cell = pCell[i];
            const int kcnt = nCount[cell];
            const int base = cell * maxN;

            for (int kk = 0; kk < kcnt; ++kk) {
                const int nb = nList[base + kk];
                int start = offsets[nb];
                int end = offsets[nb + 1];

                for (int t = start; t < end; ++t) {
                    const int j = sidx[t];
                    if (j == i) continue;

                    const float dx = px[j] - pix;
                    const float dy = py[j] - piy;
                    const float r2 = dx * dx + dy * dy;

                    if (r2 > cull2 || r2 < 1e-6f) continue;

                    const float dist = std::sqrt(r2);
                    const float invd = 1.0f / dist;
                    const float ux = dx * invd;
                    const float uy = dy * invd;

                    float dist2 = dist;
                    if (dist2 < 1e-1f) dist2 = 1e-1f;

                    const float invd2 = 1.0f / (dist2 * dist2);
                    float mag = (mi * pm[j]) * (0.01f * invd2);

                    float ftx = ux * mag;
                    float fty = uy * mag;

                    if (dist2 > cull075) {
                        float decay = 1.0f - (dist2 - cull075) * invC25;
                        ftx *= decay;
                        fty *= decay;
                    }

                    fx += ftx;
                    fy += fty;
                }
            }

            Vec2 force{fx, fy};
            out[i] = updateParticle(p[i], force, dt);
        }
    }
};

Simulator* createSimulator() {
    return new MySimulator();
}