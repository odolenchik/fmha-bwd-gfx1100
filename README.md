# 🚀 Flash Attention Backward for AMD RDNA 3 (gfx1100 / RX 7900 XT)

[![GPU](https://img.shields.io/badge/GPU-RX%207900%20XT-red)](https://www.amd.com/en/products/graphics/amd-radeon-rx-7900-xt)
[![ROCm](https://img.shields.io/badge/ROCm-7.2.1-blue)](https://www.amd.com/en/products/software/rocm.html)
[![WMMA](https://img.shields.io/badge/WMMA-Working-brightgreen)]()
[![Status](https://img.shields.io/badge/Status-Active%20Development-yellow)]()

## 📋 **О проекте**

Полная реализация **backward-прохода Flash Attention** для AMD RX 7900 XT (RDNA 3, gfx1100) с использованием **тензорных ядер WMMA** (Wave Matrix Multiply-Accumulate).

### 🎯 **Цели проекта**
1. ✅ Реализовать **математически правильный** backward pass
2. ✅ Обойти официальные ограничения Composable Kernel для gfx11
3. ✅ Интегрировать **WMMA** для максимальной производительности
4. ⏳ Достичь **30-50 TFLOPS** через LDS и конвейеризацию
5. ⏳ Интегрировать в **PyTorch** как custom extension

---

## ✅ **Что уже работает**

### 🔥 **Путь A: Правильная математика (полностью завершён)**

| Компонент | Формула | Статус | Точность |
|:---|:---|:---:|:---:|
| **dP** | `dP = dO @ Vᵀ` | ✅ | max_diff = 0 |
| **Softmax backward** | `dS = P * (dP - rowsum(dP * P))` | ✅ | max_diff = 0.0005 |
| **dQ** | `dQ = dS @ K` | ✅ | max_diff = 0.002 |
| **dK** | `dK = dSᵀ @ Q` | ✅ | max_diff = 0.001 |
| **dV** | `dV = Pᵀ @ dO` | ✅ | max_diff = 0.001 |

**Полный интеграционный тест**: `test_full_backward_correct.cpp` ✅ **PASSED**

### 🚀 **Путь B: WMMA оптимизация (в процессе)**

| Этап | Статус | Производительность |
|:---|:---:|:---:|
| Наивное скалярное ядро | ✅ | 101 us (0.33 TFLOPS) |
| **rocWMMA базовое** | ✅ | **56 us (2.4 TFLOPS)** |
| rocWMMA + LDS | ⏳ | ~10 us (15 TFLOPS) ожид. |
| rocWMMA + LDS + двойная буферизация | ⏳ | ~3 us (50 TFLOPS) ожид. |

**Базовое WMMA-ядро уже даёт 1.8× ускорение!**

---

## 📁 **Структура проекта**
fmha-bwd-gfx1100/
├── src/kernel/
│ ├── correct/ # ✅ Правильные ядра (Путь A)
│ │ ├── cpu_reference.hpp # CPU эталон
│ │ ├── fmha_bwd_dp_kernel.hpp # dP = dO @ Vᵀ
│ │ ├── fmha_bwd_softmax_kernel.hpp
│ │ ├── fmha_bwd_dq_kernel.hpp
│ │ ├── fmha_bwd_dk_kernel.hpp
│ │ └── fmha_bwd_dv_kernel.hpp
│ └── wmma/ # ⏳ WMMA ядра (Путь B)
│ └── fmha_bwd_dq_wmma.hpp
├── tests/
│ ├── test_full_backward_correct.cpp # Интеграционный тест ✅
│ ├── test_dq_rocwmma_final.cpp # WMMA dQ (работает) ✅
│ ├── test_dq_rocwmma_large.cpp # Бенчмарк 1024×1024 ✅
│ ├── test_wmma_simple.cpp # WMMA интринсик тест ✅
│ └── benchmark_comparison.cpp # Сравнение производительности
├── docs/
│ └── cmake_bypass.md # Обход CMake-фильтра gfx11
└── README.md

---

## 📊 **Производительность (M=N=1024, K=64)**

| Реализация | Время | TFLOPS | Ускорение |
|:---|:---:|:---:|:---:|
| Наивное скалярное | 101 us | 0.33 | 1.0× |
| **rocWMMA (базовое)** | **56 us** | **2.4** | **1.8×** |
| rocWMMA + LDS (в процессе) | ~10 us | ~15 | ~10× |
| rocWMMA + LDS + двойная буферизация (план) | ~3 us | ~50 | ~30× |
| **Теоретический пик RX 7900 XT** | — | **61** | — |

---

## 🔧 **Сборка и запуск**

### **Требования**
- ROCm 7.2.1+
- RX 7900 XT (или другая карта gfx1100)
- Ubuntu 24.04

### **Компиляция WMMA теста**
```bash
/opt/rocm/bin/hipcc -std=c++17 -O3 \
  --offload-arch=gfx1100 \
  -I/opt/rocm/include \
  tests/test_dq_rocwmma_large.cpp -o test_dq_rocwmma_large

./test_dq_rocwmma_large

Запуск полного backward теста
/opt/rocm/bin/hipcc -std=c++17 \
  -I. -Itests \
  --offload-arch=gfx1100 -D__HIP_PLATFORM_AMD__ \
  tests/test_full_backward_correct.cpp -o test_full_backward_correct

./test_full_backward_correct

🗺️ Roadmap
✅ Завершено

    Правильная математика backward pass

    CPU референс для верификации

    Обход CMake-фильтра gfx11 в Composable Kernel

    WMMA интринсик тест (работает на gfx1100)

    Интеграция rocWMMA для dQ

    Базовое WMMA-ядро (1.8× ускорение)

⏳ В процессе

    LDS (shared memory) для WMMA-ядра

    Двойная буферизация тайлов

    Оптимизация размеров тайлов (BM, BN, BK)

📅 Запланировано

    WMMA для dK (dSᵀ @ Q)

    WMMA для dV (Pᵀ @ dO)

    WMMA для dP (dO @ Vᵀ)

    WMMA для softmax backward

    Объединение всех компонентов в одно ядро

    Интеграция в PyTorch (torch.utils.cpp_extension)

    Поддержка dropout, causal mask, bias

📚 Ключевые инсайты
🔓 Обход CMake-фильтра gfx11

Официальный Composable Kernel не поддерживает gfx11 для FMHA backward. Решение:
cmake

# Закомментировать строку в CMakeLists.txt:
# list(FILTER INST_TARGETS INCLUDE REGEX "gfx9|gfx1[12]")

✅ WMMA работает на gfx1100

Тест test_wmma_simple.cpp доказал, что интринсик __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 полностью функционален.
🎯 rocWMMA упрощает разработку

Библиотека rocWMMA предоставляет готовые C++ обёртки для WMMA, скрывая сложность раскладки данных по регистрам.
🤝 Contributing

Pull requests приветствуются! Особенно интересны:

    Оптимизации LDS и двойной буферизации

    Реализация WMMA для dK и dV

    Интеграция в PyTorch

📄 Лицензия

MIT — свободно используйте, модифицируйте и учитесь.
🙏 Благодарности

    Composable Kernel — основа для изучения WMMA

    rocWMMA — библиотека для работы с тензорными ядрами

    Flash Attention — эталонный алгоритм

Автор: odolenchik
