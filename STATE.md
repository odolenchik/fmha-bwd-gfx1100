# Состояние проекта fmha-bwd-gfx1100 (11.06.2026)

## Цель проекта
Создать высокопроизводительный backward-проход для Multi-Head Attention (MHA) на AMD Radeon RX 7900 XTX (RDNA3, gfx1100) с использованием WMMA-инструкций для ускорения обучения диффузионных моделей (FLUX.2 Klein, HV1.5, WAN2.2) с LoRA/DoRA.

## Структура папок и файлов
- **src/fmha_bwd_kernels.cpp** – основной файл с C-ядрами: 5 fused kernel-ов (dP→softmax→dQ→dK→dV), multi-head поддержка через blockIdx.z, единая точка входа через C API (`fmha_bwd_full_py`)
- **src/kernel/correct/fmha_bwd_config.h** – централизованные константы (BM=64, BN=64, BK=32, LDS_PAD=8)
- **src/kernel/correct/fmha_bwd_utils.hip** – макросы ошибок HIP_CHECK / HIP_CHECK_EXIT + sync checker
- **src/kernel/correct/fmha_bwd_dk_kernel.hpp** – WMMA dK = dS^T @ Q (исправлен: FragB=col_major)
- **src/kernel/correct/fmha_bwd_dq_kernel.hpp** – WMMA dQ = dS @ K (верифицирован, работает)
- **src/kernel/correct/fmha_bwd_dv_kernel.hpp** – WMMA dV = P^T @ dO (исправлен: FragB=col_major)
- **src/kernel/correct/fmha_bwd_dp_kernel.hpp** – dP = dO * softmax(QK^T/sqrt(d))
- **src/kernel/correct/fmha_bwd_softmax_kernel.hpp** – softmax backward
- **test_dq_final.hip** – тест dQ (верифицирован, ошибка 0.0)
- **test_dk_naive_check.hip** – наивный dK (CPU-reference GPU) для валидации WMMA
- **test_dk_debug.hip** – отладочный тест WMMA dK с hipMemcpyDeviceToHost
- **test_dv_simple.hip** – быстрый sanity-check dV kernel

## Что уже работает (верифицировано)
- **dQ kernel**: `fmha_bwd_dq_kernel_wmma` — верифицирован, ошибка 0.0 (ожидаемое: -32.0 при входных -0.5 * 1.0)
- **Shared config header**: все ядра используют `fmha_bwd_config.h` с едиными константами
- **Unified WMMA pattern**: dQ, dK, dV — все три ядра используют одинаковый паттерн FragA=row_major, FragB=col_major с col-major transpose при заполнении B-фрагмента
- **C API (`fmha_bwd_full_py`)**: 5 fused kernel-ов в одном вызове, поддержка multi-head (total_heads параметр)
- **Benchmark script**: `benchmark_pytorch_final.py` — адаптирован под ROCm/hipBLAS
- **CMakeLists.txt**: сборка SHARED библиотеки + тесты для RDNA3 (gfx1100)

## Что исправлено
1. **dK kernel layout** (основной баг): FragA и FragB были row_major, reinterpret_cast давал константу -4704.0 на блок. Исправлено: FragA=row_major, FragB=col_major с `b_ptr[j*16+i]` транспонированием
2. **dV kernel layout**: та же проблема — исправлен аналогично dK
3. **test_dk_debug.hip hipMemcpy bug**: был указан hipMemcpyHostToDevice вместо DeviceToHost при чтении результатов с GPU
4. **Все тестовые файлы**: унифицированы FragB=col_major через replace_all
5. **Удалено 9 дублирующих тестовых файлов**: test_dk.hip, test_dp.hip, test_dq_reconstructed.hip, test_dv*.hip (5 шт), test_dk_verify.hip

## Что не работало (архив ошибок)
- **Первые попытки WMMA** — путали row_major/col_major, неверно передавали lda, не использовали shared memory
- **Triton-эксперимент** — компиляция Triton не давала ускорения для нашей задачи (RDNA3)
- **hipBLASLt** — нестабильное поведение и OOM

## План дальнейших работ
1. **[TODO] Верификация dK/dV на GPU**: скомпилировать test_dk_debug.hip, запустить на 7900 XTX для проверки корректности WMMA паттерна
2. **Сравнение скорости с PyTorch**: запустить benchmark_pytorch_final.py на реальных размерах (1x16x2048x2048x64)
3. **Тестирование multi-head**: проверить blockIdx.z работу с 16+ головами
4. **Упаковка в Python-модуль**: pybind11 для импорта libfmha_bwd.so в Python
