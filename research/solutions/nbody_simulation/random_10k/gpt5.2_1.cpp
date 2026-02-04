#include "world.h"
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

class MySimulator : public Simulator {
private:
    int numThreads = 0;
    int N = 0;

    // Grid data (rebuilt each step)
    float cellSize = 0.0f;
    float invCellSize = 0.0f;
    float originX = 0.0f, originY = 0.0f;
    int nx = 0, ny = 0;
    int numCells = 0;
    int kRange = 0;

    std::vector<int> cellX, cellY, cellId;
    std::vector<int> particleOrder;
    std::vector<int> cellStart;

    // Neighbor cell lists
    std::vector<int> neighOffsets;
    std::vector<int> neighIds;

    // Per-thread counting sort helpers
    std::vector<int> threadCounts;   // [T][numCells]
    std::vector<int> threadOffsets;  // [T][numCells] (used as write cursor in fill phase)

    static inline Vec2 computeForceFast(const Particle &target, const Particle &attractor,
                                        float cullRadius, float dx, float dy, float dist2) {
        // dist2 already checked against r^2 by caller for performance; keep full semantics.
        float dist = std::sqrt(dist2);
        if (dist < 1e-3f) return Vec2(0.0f, 0.0f);

        float invDist = 1.0f / dist;
        float dirx = dx * invDist;
        float diry = dy * invDist;

        if (dist > cullRadius) return Vec2(0.0f, 0.0f);

        if (dist < 1e-1f) dist = 1e-1f;

        const float G = 0.01f;
        float invDist2 = 1.0f / (dist * dist);
        float scalar = G * invDist2;

        // Match computeForce's multiplication order: ((dir * target.mass) * attractor.mass) * scalar
        float fx = dirx * target.mass;
        float fy = diry * target.mass;
        fx *= attractor.mass;
        fy *= attractor.mass;
        fx *= scalar;
        fy *= scalar;

        float decayStart = cullRadius * 0.75f;
        if (dist > decayStart) {
            float decay = 1.0f - (dist - decayStart) / (cullRadius * 0.25f);
            fx *= decay;
            fy *= decay;
        }
        return Vec2(fx, fy);
    }

    void ensureCapacity(int n) {
        if (N != n) {
            N = n;
            cellX.resize(N);
            cellY.resize(N);
            cellId.resize(N);
            particleOrder.resize(N);
        }
    }

    void buildNeighborLists() {
        neighOffsets.assign(numCells + 1, 0);
        neighIds.clear();
        int maxPerCell = (2 * kRange + 1) * (2 * kRange + 1);
        neighIds.reserve(numCells * maxPerCell);

        int out = 0;
        for (int c = 0; c < numCells; ++c) {
            neighOffsets[c] = out;
            int cx = c % nx;
            int cy = c / nx;
            int x0 = std::max(0, cx - kRange);
            int x1 = std::min(nx - 1, cx + kRange);
            int y0 = std::max(0, cy - kRange);
            int y1 = std::min(ny - 1, cy + kRange);
            for (int yy = y0; yy <= y1; ++yy) {
                int base = yy * nx;
                for (int xx = x0; xx <= x1; ++xx) {
                    neighIds.push_back(base + xx);
                    ++out;
                }
            }
        }
        neighOffsets[numCells] = out;
    }

public:
    void init(int numParticles, StepParameters params) override {
        (void)params;
        numThreads = omp_get_max_threads();
        if (numThreads <= 0) numThreads = 1;
        omp_set_dynamic(0);
        omp_set_num_threads(numThreads);
        ensureCapacity(numParticles);

        cellStart.clear();
        threadCounts.clear();
        threadOffsets.clear();
        neighOffsets.clear();
        neighIds.clear();

        // Reset grid params
        cellSize = 0.0f;
        invCellSize = 0.0f;
        originX = originY = 0.0f;
        nx = ny = numCells = 0;
        kRange = 0;
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        const int n = (int)particles.size();
        if (n == 0) return;
        ensureCapacity(n);
        if ((int)newParticles.size() != n) newParticles.resize(n);

        const float r = params.cullRadius;
        const float r2 = r * r;
        const float dt = params.deltaTime;

        // Compute bounds
        float minX = std::numeric_limits<float>::max();
        float minY = std::numeric_limits<float>::max();
        float maxX = -std::numeric_limits<float>::max();
        float maxY = -std::numeric_limits<float>::max();

        #pragma omp parallel for schedule(static) reduction(min:minX) reduction(min:minY) reduction(max:maxX) reduction(max:maxY)
        for (int i = 0; i < n; ++i) {
            const float x = particles[i].position.x;
            const float y = particles[i].position.y;
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
        }

        // Choose cell size (tune for cut-off)
        float s = r * 0.5f;
        if (s < 1e-3f) s = 1e-3f;

        // Keep grid size reasonable if particles wander far
        const float spanX = std::max(1e-3f, maxX - minX);
        const float spanY = std::max(1e-3f, maxY - minY);
        const int maxCells = 16384;
        while (true) {
            int tx = (int)(spanX / s) + 3;
            int ty = (int)(spanY / s) + 3;
            if ((int64_t)tx * (int64_t)ty <= maxCells) break;
            s *= 2.0f;
            if (s > r) break;
        }

        cellSize = s;
        invCellSize = 1.0f / cellSize;

        // Origin snapped down, with small epsilon to keep indices non-negative under float error
        originX = std::floor(minX * invCellSize) * cellSize - 1e-4f;
        originY = std::floor(minY * invCellSize) * cellSize - 1e-4f;

        nx = (int)((maxX - originX) * invCellSize) + 2;
        ny = (int)((maxY - originY) * invCellSize) + 2;
        if (nx < 1) nx = 1;
        if (ny < 1) ny = 1;

        numCells = nx * ny;
        if (numCells < 1) numCells = 1;

        kRange = (int)std::ceil(r / cellSize);
        if (kRange < 1) kRange = 1;

        // Build neighbor lists for this grid
        buildNeighborLists();

        // Allocate/reuse counting sort buffers
        cellStart.assign(numCells + 1, 0);
        threadCounts.assign(numThreads * numCells, 0);
        threadOffsets.assign(numThreads * numCells, 0);

        // 1) Count per cell (thread-local)
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            int *localCounts = threadCounts.data() + (size_t)tid * (size_t)numCells;

            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                const float px = particles[i].position.x;
                const float py = particles[i].position.y;

                int cx = (int)((px - originX) * invCellSize);
                int cy = (int)((py - originY) * invCellSize);
                if (cx < 0) cx = 0; else if (cx >= nx) cx = nx - 1;
                if (cy < 0) cy = 0; else if (cy >= ny) cy = ny - 1;

                cellX[i] = cx;
                cellY[i] = cy;
                int cid = cy * nx + cx;
                cellId[i] = cid;
                localCounts[cid] += 1;
            }
        }

        // 2) Reduce counts into cellStart counts (cellStart used temporarily as counts)
        std::vector<int> cellCounts(numCells, 0);
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < numCells; ++c) {
            int sum = 0;
            for (int t = 0; t < numThreads; ++t) {
                sum += threadCounts[(size_t)t * (size_t)numCells + (size_t)c];
            }
            cellCounts[c] = sum;
        }

        // 3) Prefix sum to cellStart
        cellStart[0] = 0;
        for (int c = 0; c < numCells; ++c) {
            cellStart[c + 1] = cellStart[c] + cellCounts[c];
        }

        // 4) Compute per-thread write offsets for stable fill (across threads)
        for (int c = 0; c < numCells; ++c) {
            int off = cellStart[c];
            for (int t = 0; t < numThreads; ++t) {
                size_t idx = (size_t)t * (size_t)numCells + (size_t)c;
                threadOffsets[idx] = off;
                off += threadCounts[idx];
            }
        }

        // 5) Fill particleOrder by cell
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            int *localWrite = threadOffsets.data() + (size_t)tid * (size_t)numCells;

            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                int cid = cellId[i];
                int pos = localWrite[cid]++;
                particleOrder[pos] = i;
            }
        }

        // 6) Compute forces and update particles
        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < n; ++i) {
            const Particle &pi = particles[i];
            float fx = 0.0f, fy = 0.0f;

            const int cid = cellId[i];
            const int nb0 = neighOffsets[cid];
            const int nb1 = neighOffsets[cid + 1];

            for (int nb = nb0; nb < nb1; ++nb) {
                const int nc = neighIds[nb];
                const int start = cellStart[nc];
                const int end = cellStart[nc + 1];

                for (int p = start; p < end; ++p) {
                    const int j = particleOrder[p];
                    if (j == i) continue;
                    const Particle &pj = particles[j];

                    const float dx = pj.position.x - pi.position.x;
                    const float dy = pj.position.y - pi.position.y;
                    const float dist2 = dx * dx + dy * dy;

                    if (dist2 > r2 || dist2 < 1e-6f) continue;

                    Vec2 f = computeForceFast(pi, pj, r, dx, dy, dist2);
                    fx += f.x;
                    fy += f.y;
                }
            }

            newParticles[i] = updateParticle(pi, Vec2(fx, fy), dt);
        }
    }
};

Simulator* createSimulator() {
    return new MySimulator();
}