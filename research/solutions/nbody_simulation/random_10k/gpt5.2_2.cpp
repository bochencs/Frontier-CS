#include "world.h"
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

class MySimulator : public Simulator {
private:
    int numThreads = 16;

    // Structure-of-arrays cache (per step)
    std::vector<float> posX, posY, mass;

    // Uniform grid (rebuilt each step)
    float cellSize = 1.0f;
    float invCellSize = 1.0f;
    float originX = 0.0f, originY = 0.0f;
    int gridW = 0, gridH = 0;
    std::vector<int> cellX, cellY, cellId;
    std::vector<int> cellCounts, cellStart, cellWrite;
    std::vector<int> cellParticles;

    static inline int clampInt(int v, int lo, int hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    inline void ensureSize(int n) {
        if ((int)posX.size() != n) {
            posX.resize(n);
            posY.resize(n);
            mass.resize(n);
            cellX.resize(n);
            cellY.resize(n);
            cellId.resize(n);
            cellParticles.resize(n);
        }
    }

    inline void buildSoA(const std::vector<Particle> &particles) {
        const int n = (int)particles.size();
        for (int i = 0; i < n; i++) {
            posX[i] = particles[i].position.x;
            posY[i] = particles[i].position.y;
            mass[i] = particles[i].mass;
        }
    }

    inline void buildGrid(const std::vector<Particle> &particles, float cullRadius) {
        const int n = (int)particles.size();
        if (n == 0) {
            gridW = gridH = 0;
            return;
        }

        float minX = particles[0].position.x, maxX = particles[0].position.x;
        float minY = particles[0].position.y, maxY = particles[0].position.y;
        for (int i = 1; i < n; i++) {
            const float x = particles[i].position.x;
            const float y = particles[i].position.y;
            minX = std::min(minX, x);
            maxX = std::max(maxX, x);
            minY = std::min(minY, y);
            maxY = std::max(maxY, y);
        }

        cellSize = std::max(cullRadius * 0.25f, 1.0f);
        invCellSize = 1.0f / cellSize;

        const float margin = cullRadius + cellSize;
        originX = minX - margin;
        originY = minY - margin;

        const float spanX = (maxX - minX) + 2.0f * margin;
        const float spanY = (maxY - minY) + 2.0f * margin;

        gridW = (int)(spanX * invCellSize) + 1;
        gridH = (int)(spanY * invCellSize) + 1;
        gridW = std::max(gridW, 1);
        gridH = std::max(gridH, 1);

        const int64_t M64 = (int64_t)gridW * (int64_t)gridH;
        if (M64 <= 0 || M64 > 5000000) {
            gridW = gridH = 0;
            return;
        }
        const int M = (int)M64;

        cellCounts.assign(M, 0);

        for (int i = 0; i < n; i++) {
            int ix = (int)((posX[i] - originX) * invCellSize);
            int iy = (int)((posY[i] - originY) * invCellSize);
            ix = clampInt(ix, 0, gridW - 1);
            iy = clampInt(iy, 0, gridH - 1);
            cellX[i] = ix;
            cellY[i] = iy;
            const int cid = iy * gridW + ix;
            cellId[i] = cid;
            cellCounts[cid]++;
        }

        cellStart.resize(M + 1);
        cellStart[0] = 0;
        for (int c = 0; c < M; c++) cellStart[c + 1] = cellStart[c] + cellCounts[c];

        cellWrite.assign(cellStart.begin(), cellStart.end() - 1);

        for (int i = 0; i < n; i++) {
            const int cid = cellId[i];
            const int p = cellWrite[cid]++;
            cellParticles[p] = i;
        }
    }

    static inline void addForce(float &fxSum, float &fySum,
                                float tx, float ty, float tmass,
                                float ax, float ay, float amass,
                                float cullRadius, float r2) {
        const float dx = ax - tx;
        const float dy = ay - ty;
        const float dist2 = dx * dx + dy * dy;
        if (dist2 > r2) return;
        if (dist2 < 1e-6f) return;

        const float dist = std::sqrt(dist2);
        const float invDist = 1.0f / dist;
        float dirx = dx * invDist;
        float diry = dy * invDist;

        float d = dist;
        if (d < 1e-1f) d = 1e-1f;

        float fx = dirx;
        float fy = diry;

        fx *= tmass;
        fy *= tmass;
        fx *= amass;
        fy *= amass;

        const float invd2 = 1.0f / (d * d);
        fx *= (0.01f * invd2);
        fy *= (0.01f * invd2);

        const float decayStart = cullRadius * 0.75f;
        if (d > decayStart) {
            const float decay = 1.0f - (d - decayStart) / (cullRadius * 0.25f);
            fx *= decay;
            fy *= decay;
        }

        fxSum += fx;
        fySum += fy;
    }

    inline void fallbackBrute(const std::vector<Particle> &particles,
                              std::vector<Particle> &newParticles,
                              StepParameters params) {
        const int n = (int)particles.size();
        const float r = params.cullRadius;
        const float r2 = r * r;
        #pragma omp parallel for schedule(static, 32)
        for (int i = 0; i < n; i++) {
            const Particle &pi = particles[i];
            const float tx = pi.position.x;
            const float ty = pi.position.y;
            const float tm = pi.mass;
            float fxSum = 0.0f, fySum = 0.0f;

            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                addForce(fxSum, fySum, tx, ty, tm, posX[j], posY[j], mass[j], r, r2);
            }

            Vec2 force(fxSum, fySum);
            newParticles[i] = updateParticle(pi, force, params.deltaTime);
        }
    }

public:
    void init(int numParticles, StepParameters params) override {
        (void)numParticles;
        (void)params;
        omp_set_dynamic(0);
        int maxT = omp_get_max_threads();
        numThreads = std::min(16, maxT);
        numThreads = std::max(1, numThreads);
        omp_set_num_threads(numThreads);
    }

    void simulateStep(std::vector<Particle> &particles,
                      std::vector<Particle> &newParticles,
                      StepParameters params) override {
        const int n = (int)particles.size();
        if (n == 0) return;
        if ((int)newParticles.size() != n) newParticles.resize(n);

        ensureSize(n);
        buildSoA(particles);
        buildGrid(particles, params.cullRadius);

        if (gridW == 0 || gridH == 0) {
            fallbackBrute(particles, newParticles, params);
            return;
        }

        const float r = params.cullRadius;
        const float r2 = r * r;
        const int range = (int)std::ceil(r * invCellSize);

        #pragma omp parallel for schedule(static, 32)
        for (int i = 0; i < n; i++) {
            const Particle &pi = particles[i];
            const float tx = pi.position.x;
            const float ty = pi.position.y;
            const float tm = pi.mass;

            float fxSum = 0.0f, fySum = 0.0f;

            const int ix = cellX[i];
            const int iy = cellY[i];

            const int y0 = std::max(0, iy - range);
            const int y1 = std::min(gridH - 1, iy + range);
            const int x0 = std::max(0, ix - range);
            const int x1 = std::min(gridW - 1, ix + range);

            for (int cy = y0; cy <= y1; cy++) {
                const int row = cy * gridW;
                for (int cx = x0; cx <= x1; cx++) {
                    const int cid = row + cx;
                    const int start = cellStart[cid];
                    const int end = cellStart[cid + 1];
                    for (int p = start; p < end; p++) {
                        const int j = cellParticles[p];
                        if (j == i) continue;
                        addForce(fxSum, fySum, tx, ty, tm, posX[j], posY[j], mass[j], r, r2);
                    }
                }
            }

            Vec2 force(fxSum, fySum);
            newParticles[i] = updateParticle(pi, force, params.deltaTime);
        }
    }
};

Simulator* createSimulator() {
    return new MySimulator();
}