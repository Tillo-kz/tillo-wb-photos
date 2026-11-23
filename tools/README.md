# Инструменты для 3D фигурок Tillo
Код из навыка `tillo-figurka-3d` (основа: навыки figurine-3d-print-hunyuan и bez-podderzhek).
Сборка в облаке: `g++ -O3 -o rayz rayz.cpp && g++ -O3 -o sdfband sdfband.cpp && g++ -O2 -o qem qem.cpp`
Python: numpy, scipy, scikit-image, opencv. Проверено 24.11.2026 на синтетической фигурке 15 мм: в воздухе 1.04 → 0.25 мм², ~90 с.

## bambu_template/
Настройки из проектов Tillo (P1S, сопло 0.4, 0.24mm Draft, стенки 2, заполнение 15%, без поддержек, prime tower, flush 1).
Пластина: 166–280 фигурок сеткой, зазор ≈1 мм. Цвет граней — paint_color (4, 8, 0C, 1C). Только конфиги, без моделей.
