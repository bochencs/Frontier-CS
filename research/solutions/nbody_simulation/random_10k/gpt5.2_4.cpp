#include "world.h"
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

class MySimulator : public Simulator {
private:
    int numThreads = 16;

    int N = 0;

    // Grid parameters (rebuilt each step)
    float cellSize = 1.0f;
    float invCellSize = 1.0f;
    float originX = 0.0f, originY = 0.0f;
    int gridW = 0, gridH = 0, numCells = 0;
    int neighborRange = 1;

    // Persistent buffers
    std::vector<float> posX, posY, mass;
    std::vector<int> cellIdOfParticle;
    std::vector<int> particleOrder;

    std::vector<int> cellCounts;
    std::vector<int> cellOffsets;

    std::vector<int> threadCounts;
    std::vector<int> threadPos;

    static inline void accumulateForceFast(
        float tx, float ty, float tm,
        float ax, float ay, float am,
        float cullRadius, float cullRadius2,
        float &fx, float &fy)
    {
        float dx = ax - tx;
        float dy = ay - ty;
        float dist2 = dx * dx + dy * dy;

        // Baseline uses: length() < cullRadius (strict)
        if (dist2 >= cullRadius2) return;

        // computeForce returns 0 if dist < 1e-3
        if (dist2 < 1e-6f) return;

        float dist = std::sqrt(dist2);
        if (dist < 1e-3f) return;

        float invDist = 1.0f / dist;
        float dirx = dx * invDist;
        float diry = dy * invDist;

        float distEff = dist;
        if (distEff < 1e-1f) distEff = 1e-1f;

        float scale = tm;
        scale *= am;
        scale *= (0.01f / (distEff * distEff));

        float decayStart = cullRadius * 0.75f;
        if (distEff > decayStart) {
            float decay = 1.0f - (distEff - decayStart) / (cullRadius * 0.25f);
            scale *= decay;
        }

        fx += dirx * scale;
        fy += diry * scale;
    }

    inline void ensureCapacity(int n) {
        if ((int)posX.size() < n) {
            posX.resize(n);
            posY.resize(n);
            mass.resize(n);
            cellIdOfParticle.resize(n);
            particleOrder.resize(n);
        }
    }

    void buildGrid(const StepParameters &params) {
        const float r = params.cullRadius;

        float minX = std::numeric_limits<float>::infinity();
        float minY = std::numeric_limits<float>::infinity();
        float maxX = -std::numeric_limits<float>::infinity();
        float maxY = -std::numeric_limits<float>::infinity();

        for (int i = 0; i < N; i++) {
            float x = posX[i];
            float y = posY[i];
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
        }

        // Choose smaller cells to reduce candidate neighbors; tuned for cullRadius=25 in 100x100.
        float targetCell = r * 0.25f;
        if (targetCell < 1e-3f) targetCell = 1e-3f;
        cellSize = targetCell;
        invCellSize = 1.0f / cellSize;

        neighborRange = (int)std::ceil(r * invCellSize);

        // Add margin so clamping is rare and neighbor lookups remain valid.
        originX = minX - r - cellSize;
        originY = minY - r - cellSize;
        float maxBoundX = maxX + r + cellSize;
        float maxBoundY = maxY + r + cellSize;

        float spanX = maxBoundX - originX;
        float spanY = maxBoundY - originY;

        gridW = (int)(spanX * invCellSize) + 2;
        gridH = (int)(spanY * invCellSize) + 2;
        if (gridW < 1) gridW = 1;
        if (gridH < 1) gridH = 1;

        // Safety clamp: avoid pathological huge grids if something goes wrong.
        // If triggered, enlarge cell size to reduce numCells.
        const int MAX_DIM = 8192;
        if (gridW > MAX_DIM || gridH > MAX_DIM) {
            float scale = std::max((float)gridW / MAX_DIM, (float)gridH / MAX_DIM);
            cellSize *= scale;
            invCellSize = 1.0f / cellSize;
            neighborRange = (int)std::ceil(r * invCellSize);

            gridW = (int)(spanX * invCellSize) + 2;
            gridH = (int)(spanY * invCellSize) + 2;
            if (gridW < 1) gridW = 1;
            if (gridH < 1) gridH = 1;
            if (gridW > MAX_DIM) gridW = MAX_DIM;
            if (gridH > MAX_DIM) gridH = MAX_DIM;
        }

        numCells = gridW * gridH;

        cellCounts.assign(numCells, 0);
        cellOffsets.assign(numCells + 1, 0);

        threadCounts.assign(numThreads * numCells, 0);
        threadPos.assign(numThreads * numCells, 0);

        // Parallel counting
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int *tcounts = threadCounts.data() + tid * numCells;

            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                float x = posX[i];
                float y = posY[i];

                int cx = (int)((x - originX) * invCellSize);
                int cy = (int)((y - originY) * invCellSize);

                if (cx < 0) cx = 0;
                else if (cx >= gridW) cx = gridW - 1;
                if (cy < 0) cy = 0;
                else if (cy >= gridH) cy = gridH - 1;

                int cell = cx + cy * gridW;
                cellIdOfParticle[i] = cell;
                tcounts[cell] += 1;
            }
        }

        // Reduce counts
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < numCells; c++) {
            int sum = 0;
            const int base = c;
            for (int t = 0; t < numThreads; t++) {
                sum += threadCounts[t * numCells + base];
            }
            cellCounts[c] = sum;
        }

        // Prefix sum (serial, small)
        int running = 0;
        cellOffsets[0] = 0;
        for (int c = 0; c < numCells; c++) {
            running += cellCounts[c];
            cellOffsets[c + 1] = running;
        }

        // Compute thread start positions per cell
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < numCells; c++) {
            int base = cellOffsets[c];
            int acc = 0;
            for (int t = 0; t < numThreads; t++) {
                int idx = t * numCells + c;
                threadPos[idx] = base + acc;
                acc += threadCounts[idx];
            }
        }

        // Fill particleOrder
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int *tpos = threadPos.data() + tid * numCells;

            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                int c = cellIdOfParticle[i];
                int p = tpos[c]++;
                particleOrder[p] = i;
            }
        }
    }

public:
    void init(int numParticles, StepParameters params) override {
        N = numParticles;
        int procs = omp_get_num_procs();
        numThreads = std::min(16, std::max(1, procs));
        omp_set_num_threads(numThreads);
        ensureCapacity(N);
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override
    {
        N = (int)particles.size();
        ensureCapacity(N);
        if ((int)newParticles.size() != N) newParticles.resize(N);

        // Extract SoA for fast neighbor reads
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            posX[i] = particles[i].position.x;
            posY[i] = particles[i].position.y;
            mass[i] = particles[i].mass;
        }

        buildGrid(params);

        const float r = params.cullRadius;
        const float r2 = r * r;
        const float dt = params.deltaTime;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            float tx = posX[i];
            float ty = posY[i];
            float tm = mass[i];

            int cell = cellIdOfParticle[i];
            int cx = cell % gridW;
            int cy = cell / gridW;

            float fx = 0.0f, fy = 0.0f;

            int y0 = cy - neighborRange;
            int y1 = cy + neighborRange;
            int x0 = cx - neighborRange;
            int x1 = cx + neighborRange;

            if (y0 < 0) y0 = 0;
            if (x0 < 0) x0 = 0;
            if (y1 >= gridH) y1 = gridH - 1;
            if (x1 >= gridW) x1 = gridW - 1;

            for (int ny = y0; ny <= y1; ny++) {
                int rowBase = ny * gridW;
                for (int nx = x0; nx <= x1; nx++) {
                    int nc = rowBase + nx;
                    int begin = cellOffsets[nc];
                    int end = cellOffsets[nc + 1];

                    for (int k = begin; k < end; k++) {
                        int j = particleOrder[k];
                        if (j == i) continue;

                        accumulateForceFast(
                            tx, ty, tm,
                            posX[j], posY[j], mass[j],
                            r, r2, fx, fy
                        );
                    }
                }
            }

            // Inline updateParticle to reduce overhead (matches updateParticle behavior/order)
            Particle res = particles[i];
            float invMassDt = dt / res.mass;
            res.velocity.x += fx * invMassDt;
            res.velocity.y += fy * invMassDt;
            res.position.x += res.velocity.x * dt;
            res.position.y += res.velocity.y * dt;

            newParticles[i] = res;
        }
    }
};

Simulator* createSimulator() {
    return new MySimulator();
}