# Инструменты для 3D фигурок Tillo
Код из навыка `tillo-figurka-3d` (основа: навыки figurine-3d-print-hunyuan и bez-podderzhek).
Сборка в облаке: `g++ -O3 -o rayz rayz.cpp && g++ -O3 -o sdfband sdfband.cpp && g++ -O2 -o qem qem.cpp`
Python: numpy, scipy, scikit-image, opencv. Проверено 24.11.2026 на синтетической фигурке 15 мм: в воздухе 1.04 → 0.25 мм², ~90 с.
