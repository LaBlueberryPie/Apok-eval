# Third-party notices

apok-eval is released under the MIT licence (`LICENSE`). Parts of it are ported from, or modelled on, the
projects below, which are also MIT-licensed. Their notices are reproduced here as their licences require.

## ScrollPrize/villa

- `src/apok_eval/eval/fibres.py` ports villa's VC3D fibre-trace reading from
  `vesuvius/src/vc3d_fiber_format/__init__.py` (`parse_vc3d_fiber_format`) and
  `spiral-fitting/spiral_helpers.py` (`load_fiber_point_collection`) at commit `e0bbb8b`: the trim to the
  control-point span, the dyadic scale and the H/V family rule.
- Other modules cite villa's Python code by `path:line` to document the formats they read. That code was read,
  not copied.
- villa's `volume-cartographer/` directory (VC3D) is licensed separately under GPL-3.0. apok-eval contains no
  VC3D code: its C++ sources were read only to learn the file formats, and are cited as such.

```
MIT License

Copyright (c) 2024 Vesuvius Challenge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Nicodol/spiralcheck

- The idea of a geometric leakage audit (`src/apok_eval/eval/leakage.py`) is spiralcheck's. The vectorised
  point-to-triangle routine there follows Ericson, *Real-Time Collision Detection* §5.1.5, in the same masked,
  case-by-case form as spiralcheck's `closest_point_on_triangles` (`src/spiralcheck/geometry.py`).

```
MIT License

Copyright (c) 2026 Nicolas Dolegieviez (Nicodol)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
