#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>   
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

py::dict run_simulation(py::array_t<double, py::array::c_style | py::array::forcecast> pos_array, 
                        py::array_t<double, py::array::c_style | py::array::forcecast> vel_array, 
                        int M, int N, double L, double m, double dt, 
                        double radius, int steps, int k_pairs, double target_ke) {
    
    auto pos = pos_array.mutable_unchecked<3>();
    auto vel = vel_array.mutable_unchecked<3>();

    std::vector<double> time_data(steps, 0.0);
    std::vector<double> ensemble_pressure_data(steps, 0.0);
    std::vector<double> ke_data(steps, 0.0);
    std::vector<double> mean_abs_px_data(steps, 0.0);
    std::vector<double> mean_abs_py_data(steps, 0.0);

    double diam_sq = (2 * radius) * (2 * radius);
    
    // Hücre Tabanlı (Spatial Hashing) Ağ Kurulumu
    double cell_size = 2.0 * radius * 3.0;
    int n_cells_x = std::max(1, static_cast<int>(L / cell_size));
    int n_cells_y = std::max(1, static_cast<int>(L / cell_size));
    int total_cells = n_cells_x * n_cells_y;
    double cs_x = L / n_cells_x;
    double cs_y = L / n_cells_y;

    {
        py::gil_scoped_release release;

        for (int step = 0; step < steps; ++step) {
            time_data[step] = (step + 1) * dt;

            double total_P = 0.0;
            double total_KE = 0.0;
            double total_px = 0.0;
            double total_py = 0.0;

            #pragma omp parallel
            {
                // Her Thread için Linked-List Cell (Spatial Hash) bellek tahsisi
                // (Döngü içinde malloc/new yapmamak için thread bazlı tutulur)
                std::vector<int> head(total_cells, -1);
                std::vector<int> next_node(N, -1);

                double local_P = 0.0;
                double local_KE = 0.0;
                double local_px = 0.0;
                double local_py = 0.0;

                #pragma omp for
                for (int u = 0; u < M; ++u) {
                    double wall_impulse = 0.0;

                    // 1. HAREKET VE DUVAR YANSIMALARI
                    for (int i = 0; i < N; ++i) {
                        double px = pos(u, i, 0) + vel(u, i, 0) * dt;
                        double py = pos(u, i, 1) + vel(u, i, 1) * dt;
                        double vx = vel(u, i, 0);
                        double vy = vel(u, i, 1);

                        if (px <= radius) { vx = std::abs(vx); wall_impulse += 2 * m * vx; px = radius; }
                        else if (px >= L - radius) { vx = -std::abs(vx); wall_impulse += 2 * m * std::abs(vx); px = L - radius; }

                        if (py <= radius) { vy = std::abs(vy); wall_impulse += 2 * m * vy; py = radius; }
                        else if (py >= L - radius) { vy = -std::abs(vy); wall_impulse += 2 * m * std::abs(vy); py = L - radius; }

                        pos(u, i, 0) = px; pos(u, i, 1) = py;
                        vel(u, i, 0) = vx; vel(u, i, 1) = vy;
                    }

                    local_P += wall_impulse / (dt * 4 * L);

                    // 2. MOLEKÜLER DİNAMİK (MD): HÜCRE TABANLI ÇARPIŞMALAR (O(N) Karmaşıklık)
                    std::fill(head.begin(), head.end(), -1);
                    std::fill(next_node.begin(), next_node.end(), -1);

                    // Parçacıkları grid'e yerleştir (Linked-list oluştur)
                    for (int i = 0; i < N; ++i) {
                        int cx = std::max(0, std::min(n_cells_x - 1, static_cast<int>(pos(u, i, 0) / cs_x)));
                        int cy = std::max(0, std::min(n_cells_y - 1, static_cast<int>(pos(u, i, 1) / cs_y)));
                        int cid = cx * n_cells_y + cy;
                        
                        next_node[i] = head[cid];
                        head[cid] = i;
                    }

                    // Hücreleri ve komşularını tarayarak çarpışmaları kontrol et
                    for (int cx = 0; cx < n_cells_x; ++cx) {
                        for (int cy = 0; cy < n_cells_y; ++cy) {
                            int cid = cx * n_cells_y + cy;
                            int i = head[cid];
                            
                            while (i != -1) {
                                // Komşu 9 hücreyi tara (Kendisi dahil)
                                for (int dcx = -1; dcx <= 1; ++dcx) {
                                    for (int dcy = -1; dcy <= 1; ++dcy) {
                                        int ncx = cx + dcx;
                                        int ncy = cy + dcy;
                                        
                                        if (ncx >= 0 && ncx < n_cells_x && ncy >= 0 && ncy < n_cells_y) {
                                            int ncid = ncx * n_cells_y + ncy;
                                            int j = head[ncid];
                                            
                                            while (j != -1) {
                                                if (i < j) { // Çiftleri sadece bir kere hesapla (i < j)
                                                    double dx = pos(u, j, 0) - pos(u, i, 0);
                                                    double dy = pos(u, j, 1) - pos(u, i, 1);
                                                    double dist_sq = dx * dx + dy * dy;

                                                    if (dist_sq < diam_sq && dist_sq > 1e-10) {
                                                        double dist = std::sqrt(dist_sq);
                                                        double nx = dx / dist;
                                                        double ny = dy / dist;

                                                        double dvx = vel(u, j, 0) - vel(u, i, 0);
                                                        double dvy = vel(u, j, 1) - vel(u, i, 1);
                                                        double dot = dvx * nx + dvy * ny;

                                                        if (dot < 0) { 
                                                            vel(u, i, 0) += dot * nx;
                                                            vel(u, i, 1) += dot * ny;
                                                            vel(u, j, 0) -= dot * nx;
                                                            vel(u, j, 1) -= dot * ny;

                                                            // İç içe geçme düzeltmesi
                                                            double overlap = 2.0 * radius - dist;
                                                            pos(u, i, 0) -= overlap * nx * 0.5;
                                                            pos(u, i, 1) -= overlap * ny * 0.5;
                                                            pos(u, j, 0) += overlap * nx * 0.5;
                                                            pos(u, j, 1) += overlap * ny * 0.5;
                                                        }
                                                    }
                                                }
                                                j = next_node[j];
                                            }
                                        }
                                    }
                                }
                                i = next_node[i];
                            }
                        }
                    }

                    // 3. ENERJİ HESABI VE ÖLÇEKLEME
                    double current_ke = 0.0;
                    for (int i = 0; i < N; ++i) {
                        current_ke += 0.5 * m * (vel(u, i, 0) * vel(u, i, 0) + vel(u, i, 1) * vel(u, i, 1));
                    }
                    
                    local_KE += current_ke;

                    double scale = std::sqrt(target_ke / std::max(current_ke, 1e-10));
                    double abs_px = 0.0;
                    double abs_py = 0.0;

                    for (int i = 0; i < N; ++i) {
                        vel(u, i, 0) *= scale;
                        vel(u, i, 1) *= scale;
                        abs_px += m * vel(u, i, 0);
                        abs_py += m * vel(u, i, 1);
                    }
                    
                    local_px += std::abs(abs_px);
                    local_py += std::abs(abs_py);
                }

                #pragma omp atomic
                total_P += local_P;
                #pragma omp atomic
                total_KE += local_KE;
                #pragma omp atomic
                total_px += local_px;
                #pragma omp atomic
                total_py += local_py;
            }

            ensemble_pressure_data[step] = total_P / M;
            ke_data[step] = total_KE / M;
            mean_abs_px_data[step] = total_px / M;
            mean_abs_py_data[step] = total_py / M;
        }
    } 

    py::dict results;
    results["time_data"] = py::cast(time_data);
    results["ensemble_pressure"] = py::cast(ensemble_pressure_data);
    results["ke_data"] = py::cast(ke_data);
    results["mean_abs_px"] = py::cast(mean_abs_px_data);
    results["mean_abs_py"] = py::cast(mean_abs_py_data);
    
    return results;
}

PYBIND11_MODULE(fast_ensemble, m) {
    m.def("run_simulation", &run_simulation, "Run the OpenMP accelerated gas ensemble");
}