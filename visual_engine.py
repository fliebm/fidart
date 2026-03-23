"""visual_engine.py — 3D GPU visual engine for fidart (moderngl).

Three GPU-rendered immersive modes driven by 3D fiducial positions:

  FLOW          — 30 000 point-sprite particles in 3D world space.
                  Perspective projection makes closer particles larger/brighter.
                  Additive trail accumulation → long-exposure light-painting.

  INTERFERENCE  — Full-screen fragment shader: 3D spherical waves from each
                  fiducial at its actual Z depth. Z offset shifts hue → moiré
                  gains depth that 2-D circles never had.

  CONSTELLATION — Voronoi depth zones + Delaunay aurora ribbons + nebula cloud.

All rendering goes through moderngl (OpenGL 3.3 core). The pygame window is
opened with OPENGL | DOUBLEBUF flags; moderngl attaches to that context.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pygame
import moderngl

# ── world-space constants ──────────────────────────────────────────────────────
N_PARTICLES  = 30_000
GRID_W       = 80
GRID_H       = 45
TRAIL_DECAY  = 0.935
INTERF_DECAY = 0.72
DRAG         = 0.96
DRAG_Z       = 0.985
PARTICLE_SPD = 0.0044    # world units/frame  (≈ 2.8 px at 1280 px)
FIELD_GAIN   = 4.4e-4
MAX_AGE      = 500

# World extents: X ∈ [-WX,WX], Y ∈ [-WY,WY], Z ∈ [0,WZ]
WX, WY, WZ = 1.0, 0.5625, 2.0

FOV_Y_DEG = 60.0
NEAR, FAR  = 0.1, 6.0

_WAVES = [
    (1.00,  1.10,  0.70, 0.050, 0.000),
    (0.80, -0.85,  1.30, 0.070, 1.100),
    (0.60,  1.70, -0.95, 0.040, 2.300),
    (0.40, -1.45,  1.80, 0.090, 4.100),
    (0.30,  2.30,  0.55, 0.060, 0.700),
    (0.20, -0.65, -2.10, 0.110, 3.200),
    (0.15,  3.10, -1.45, 0.080, 5.500),
    (0.10,  0.90,  3.00, 0.130, 1.900),
]

# ── colour helpers ─────────────────────────────────────────────────────────────

def _hsv_rgb_vec(h, s, v):
    h6 = np.mod(h, 1.0) * 6.0
    i  = h6.astype(np.int32) % 6
    f  = (h6 - np.floor(h6)).astype(np.float32)
    p  = (v * (1.0 - s)).astype(np.float32)
    q  = (v * (1.0 - s * f)).astype(np.float32)
    t_ = (v * (1.0 - s * (1.0 - f))).astype(np.float32)
    vv = v.astype(np.float32)
    ar = np.arange(len(i))
    r  = np.stack([vv, q,  p,  p, t_, vv])[i, ar]
    g  = np.stack([t_, vv, vv, q,  p,  p])[i, ar]
    b  = np.stack([p,  p, t_, vv, vv,  q])[i, ar]
    return r * 255.0, g * 255.0, b * 255.0


def _hue_to_rgb_f(h: float) -> Tuple[float, float, float]:
    s, v = 0.90, 1.0
    h6 = (h % 1.0) * 6.0
    i  = int(h6) % 6
    f  = h6 - int(h6)
    p, q, t_ = v*(1-s), v*(1-s*f), v*(1-s*(1-f))
    return [(v,t_,p),(q,v,p),(p,v,t_),(p,q,v),(t_,p,v),(v,p,q)][i]


# ── projection ────────────────────────────────────────────────────────────────

def _build_mvp(width: int, height: int) -> bytes:
    """MVP matrix (column-major float32) for world → clip space.

    Camera at world (0, 0, -1) looking toward +Z.
    In view space (OpenGL -Z forward): view_z = -(world_z + 1).
    """
    # View: translate z by -1 then flip Z
    V = np.array([
        [1, 0,  0,  0],
        [0, 1,  0,  0],
        [0, 0, -1,  0],
        [0, 0, -1,  1],   # last col: view_z = -world_z - 1
    ], dtype=np.float32)

    f      = 1.0 / math.tan(math.radians(FOV_Y_DEG) / 2.0)
    aspect = width / height
    nf     = 1.0 / (NEAR - FAR)
    P = np.array([
        [f/aspect,  0,               0,                    0],
        [0,         f,               0,                    0],
        [0,         0,  (FAR+NEAR)*nf,   2*FAR*NEAR*nf        ],
        [0,         0,              -1,                    0],
    ], dtype=np.float32)

    return (P @ V).T.astype(np.float32).tobytes()   # column-major for GL


def world_to_screen(wx: float, wy: float, wz: float,
                    width: int, height: int) -> Tuple[float, float]:
    """CPU-side projection matching the GPU vertex shader."""
    f      = 1.0 / math.tan(math.radians(FOV_Y_DEG) / 2.0)
    aspect = width / height
    clip_w = wz + 1.0
    if clip_w < 1e-4:
        return width / 2.0, height / 2.0
    ndc_x = (f / aspect) * wx / clip_w
    ndc_y = f * wy / clip_w
    sx = (ndc_x + 1.0) * 0.5 * width
    sy = (1.0 - ndc_y) * 0.5 * height
    return sx, sy


# ── GLSL shaders ──────────────────────────────────────────────────────────────

_VERT_QUAD = """
#version 330 core
in  vec2 a_pos;
out vec2 v_uv;
void main() {
    v_uv        = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

_FRAG_DECAY = """
#version 330 core
uniform sampler2D u_trail;
uniform float     u_decay;
in  vec2 v_uv;
out vec4 frag;
void main() { frag = texture(u_trail, v_uv) * u_decay; }
"""

_GLSL_HSV = """
vec3 hsv2rgb(float h, float s, float v) {
    float h6 = mod(h, 1.0) * 6.0;
    int   i  = int(h6);
    float f  = h6 - float(i);
    float p  = v*(1.0-s), q = v*(1.0-s*f), t = v*(1.0-s*(1.0-f));
    if (i==0) return vec3(v,t,p); if (i==1) return vec3(q,v,p);
    if (i==2) return vec3(p,v,t); if (i==3) return vec3(p,q,v);
    if (i==4) return vec3(t,p,v); return vec3(v,p,q);
}
"""

_VERT_PARTICLE = """
#version 330 core
uniform mat4  u_mvp;
uniform float u_brightness;
in vec3  a_pos;
in float a_hue;
in float a_bri;
in float a_size;
out vec3 v_color;
""" + _GLSL_HSV + """
void main() {
    gl_Position  = u_mvp * vec4(a_pos, 1.0);
    float depth  = max(gl_Position.w, 0.1);
    gl_PointSize = a_size * 10.0 / depth;
    v_color      = hsv2rgb(a_hue, 0.88, a_bri * u_brightness);
}
"""

_FRAG_PARTICLE = """
#version 330 core
in  vec3 v_color;
out vec4 frag;
void main() {
    vec2  uv = gl_PointCoord * 2.0 - 1.0;
    float d  = dot(uv, uv);
    if (d > 1.0) discard;
    frag = vec4(v_color * exp(-d * 3.0), 1.0);
}
"""

_VERT_SPHERE = """
#version 330 core
uniform mat4  u_mvp;
uniform float u_sub_bass;
uniform float u_beat;
in vec3  a_pos;
in float a_hue;
out vec3  v_color;
out float v_saturation;
""" + _GLSL_HSV + """
void main() {
    gl_Position  = u_mvp * vec4(a_pos, 1.0);
    float depth  = max(gl_Position.w, 0.1);
    float base   = 32.0 + u_sub_bass * 22.0 + u_beat * 10.0;
    gl_PointSize = base / depth;
    v_color      = hsv2rgb(a_hue, 0.82, 1.0);
    v_saturation = 0.82;
}
"""

_FRAG_SPHERE = """
#version 330 core
in  vec3  v_color;
in  float v_saturation;
out vec4  frag;
void main() {
    vec2  uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;
    float nz    = sqrt(1.0 - r2);
    vec3  n     = vec3(uv.x, -uv.y, nz);
    vec3  L1    = normalize(vec3(-0.45, 0.75, 0.50));
    float diff1 = max(dot(n, L1), 0.0);
    vec3  L2    = normalize(vec3(0.60, -0.30, 0.75));
    float diff2 = max(dot(n, L2), 0.0) * 0.30;
    vec3  H     = normalize(L1 + vec3(0.0, 0.0, 1.0));
    float spec  = pow(max(dot(n, H), 0.0), 64.0) * 0.85;
    float rim   = pow(1.0 - nz, 3.5) * 0.55;
    vec3  col   = v_color * (0.10 + diff1 * 0.80 + diff2) + vec3(spec) + v_color * rim;
    float alpha = smoothstep(1.0, 0.88, r2);
    frag = vec4(clamp(col, 0.0, 1.0), alpha);
}
"""

_FRAG_INTERF = """
#version 330 core
uniform float u_time;
uniform float u_global_hue;
uniform float u_awake;
uniform vec3  u_fid_pos[12];
uniform float u_fid_energy[12];
uniform int   u_n_fids;
in  vec2 v_uv;
out vec4 frag;
""" + _GLSL_HSV + """
void main() {
    // No people → black.  No ambient animation without presence.
    if (u_n_fids == 0) { frag = vec4(0.0, 0.0, 0.0, 1.0); return; }

    vec2 wxy = (v_uv * 2.0 - 1.0) * vec2(1.0, 0.5625);
    float wave_sum = 0.0, weight_sum = 0.0, depth_hue = 0.0;
    for (int i = 0; i < 12; i++) {
        if (i >= u_n_fids) break;
        float e    = u_fid_energy[i];
        float dz   = (u_fid_pos[i].z - 1.0) * 0.5;
        float dist = length(vec3(wxy - u_fid_pos[i].xy, dz)) * 350.0;
        float w    = e * 1.4 + 0.25;
        // Tighter spatial falloff → smaller blobs, clear dark space between them
        wave_sum   += sin(dist*0.028 - u_time*2.8) * exp(-dist*0.004) * w;
        weight_sum += w;
        depth_hue  += u_fid_pos[i].z * w;
    }
    wave_sum  /= max(weight_sum, 0.001);
    depth_hue /= max(weight_sum, 0.001);

    // Small hue shift → one dominant colour at a time, not a rainbow
    float hue = mod(wave_sum*0.08 + u_global_hue + depth_hue*0.06, 1.0);
    // No ambient floor: dark areas are truly dark
    float shaped = smoothstep(0.0, 0.55, abs(wave_sum));
    float bri    = shaped * u_awake * 0.90;
    frag = vec4(hsv2rgb(hue, 0.88, bri), 1.0);
}
"""

_FRAG_BLOOM_THRESH = """
#version 330 core
uniform sampler2D u_scene;
uniform float     u_threshold;
in  vec2 v_uv;
out vec4 frag;
void main() {
    vec3  c   = texture(u_scene, v_uv).rgb;
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    frag = vec4(lum > u_threshold ? c : vec3(0.0), 1.0);
}
"""

_FRAG_BLUR_H = """
#version 330 core
uniform sampler2D u_tex;
uniform vec2      u_texel;
in  vec2 v_uv;
out vec4 frag;
const float W[5] = float[](0.2270, 0.1945, 0.1216, 0.0540, 0.0162);
void main() {
    vec4 c = texture(u_tex, v_uv) * W[0];
    for (int i = 1; i < 5; i++) {
        c += texture(u_tex, v_uv + u_texel*float(i)) * W[i];
        c += texture(u_tex, v_uv - u_texel*float(i)) * W[i];
    }
    frag = c;
}
"""

_FRAG_BLUR_V = """
#version 330 core
uniform sampler2D u_tex;
uniform vec2      u_texel;
in  vec2 v_uv;
out vec4 frag;
const float W[5] = float[](0.2270, 0.1945, 0.1216, 0.0540, 0.0162);
void main() {
    vec4 c = texture(u_tex, v_uv) * W[0];
    for (int i = 1; i < 5; i++) {
        c += texture(u_tex, v_uv + u_texel*float(i)) * W[i];
        c += texture(u_tex, v_uv - u_texel*float(i)) * W[i];
    }
    frag = c;
}
"""

_FRAG_COMPOSITE = """
#version 330 core
uniform sampler2D u_flow;
uniform sampler2D u_interf;
uniform sampler2D u_bloom;
uniform sampler2D u_energy;
uniform float     u_blend;
uniform float     u_bloom_str;
in  vec2 v_uv;
out vec4 frag;
void main() {
    vec3  flow   = texture(u_flow,   v_uv).rgb;
    vec3  interf = texture(u_interf, v_uv).rgb;
    vec3  bloom  = texture(u_bloom,  v_uv).rgb;
    float energy = texture(u_energy, v_uv).r;
    vec3  base   = mix(flow, interf, u_blend) + bloom * u_bloom_str;
    // Low floor: when no fiducials (energy map = 0) the scene is nearly black
    base *= (0.08 + energy * 0.92);
    frag = vec4(clamp(base, 0.0, 1.0), 1.0);
}
"""

# 2-D overlay shader (screen-space NDC, no MVP)
_VERT_OVERLAY = """
#version 330 core
in  vec2 a_pos;
in  vec4 a_color;
out vec4 v_color;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_color     = a_color;
}
"""

_FRAG_OVERLAY = """
#version 330 core
in  vec4 v_color;
out vec4 frag;
void main() { frag = v_color; }
"""

_FRAG_HUD = """
#version 330 core
uniform sampler2D u_tex;
in  vec2 v_uv;
out vec4 frag;
void main() { frag = texture(u_tex, v_uv); }
"""

_FRAG_VORONOI = """
#version 330 core
uniform sampler2D u_tex;
in  vec2 v_uv;
out vec4 frag;
void main() { frag = vec4(texture(u_tex, v_uv).rgb, 1.0); }
"""

# Image overlay — desaturated, tinted, very faint
_FRAG_IMAGE = """
#version 330 core
uniform sampler2D u_tex;
uniform float     u_alpha;    // 0..1 overall opacity (max ~0.10 in practice)
uniform float     u_global_hue;
in  vec2 v_uv;
out vec4 frag;
""" + _GLSL_HSV + """
void main() {
    vec3  col = texture(u_tex, v_uv).rgb;
    // Desaturate almost completely — just structure, no original colour
    float lum = dot(col, vec3(0.299, 0.587, 0.114));
    // Tint subtly toward the current scene hue
    vec3  tint = hsv2rgb(u_global_hue, 0.25, 1.0);
    col = mix(vec3(lum), tint * lum, 0.40);
    frag = vec4(col, u_alpha);
}
"""


# Ghost constellation overlay — sampled at full alpha, composited with scissor rect
_FRAG_GHOST = """
#version 330 core
uniform sampler2D u_ghost;
uniform float     u_alpha;
in  vec2 v_uv;
out vec4 frag;
void main() {
    vec3 c = texture(u_ghost, v_uv).rgb;
    frag = vec4(c, u_alpha);
}
"""


# ── Flow Field ─────────────────────────────────────────────────────────────────

class FlowField3D:
    """Same sine-wave + vortex field as before, indexed in normalised world space.
    A lightweight analytic Z field adds gentle depth-breathing."""

    def __init__(self, grid_w: int = GRID_W, grid_h: int = GRID_H) -> None:
        self.W, self.H = grid_w, grid_h
        x = np.linspace(0, 2*math.pi, grid_w, dtype=np.float32)
        y = np.linspace(0, 2*math.pi, grid_h, dtype=np.float32)
        self._GX, self._GY = np.meshgrid(x, y)
        xi = np.arange(grid_w, dtype=np.float32)
        yi = np.arange(grid_h, dtype=np.float32)
        self._IX, self._IY = np.meshgrid(xi, yi)
        self.field = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
        self._t = 0.0

    def update(self, t: float, fiducials_grid: list) -> None:
        self._t = t
        angle = np.zeros((self.H, self.W), dtype=np.float32)
        for amp, kx, ky, omega, phase in _WAVES:
            angle += amp * np.sin(kx*self._GX + ky*self._GY + omega*t + phase)
        self.field[:, :, 0] = np.cos(angle)
        self.field[:, :, 1] = np.sin(angle)

        for entry in fiducials_grid:
            gx, gy, strength = entry[0], entry[1], entry[2]
            vdx    = entry[3] if len(entry) > 3 else 0.0
            vdy    = entry[4] if len(entry) > 3 else 0.0
            energy = entry[5] if len(entry) > 3 else 0.0
            spin   = entry[6] if len(entry) > 3 else 1.0
            dx    = self._IX - gx
            dy    = self._IY - gy
            dist2 = dx*dx + dy*dy + 0.15
            vortex_mix = max(0.10, 1.0 - energy * 0.85)
            vs = strength * 9.0 * vortex_mix / dist2
            self.field[:, :, 0] -= dy * vs * spin
            self.field[:, :, 1] += dx * vs * spin
            if energy > 0.05:
                local = np.exp(-dist2 * 0.22)
                dir_str = energy * 1.3 * strength
                self.field[:, :, 0] += vdx * dir_str * local * 7.0
                self.field[:, :, 1] += vdy * dir_str * local * 7.0
            ps = strength * 2.5 * (1.0 - energy * 0.6) / (dist2 + 1.5)
            self.field[:, :, 0] -= dx * ps
            self.field[:, :, 1] -= dy * ps

        mag = np.hypot(self.field[:, :, 0], self.field[:, :, 1]) + 1e-6
        self.field[:, :, 0] /= mag
        self.field[:, :, 1] /= mag

    def z_velocity(self, px: np.ndarray, py: np.ndarray,
                   pz: np.ndarray) -> np.ndarray:
        """Analytic Z field — slow depth breathing, no grid needed."""
        return (np.sin(px * 2.3 + pz * 1.5 + self._t * 0.08) * 0.18
              + np.sin(py * 3.1 + pz * 0.7 + self._t * 0.12) * 0.12
               ).astype(np.float32)


# ── Particle System ────────────────────────────────────────────────────────────

class ParticleSystem3D:
    """30 000 particles in 3D world space. VBO layout: [px py pz hue bri size]."""

    def __init__(self, n: int = N_PARTICLES) -> None:
        self.n   = n
        rng      = np.random.default_rng(1)
        self.px  = rng.uniform(-WX, WX, n).astype(np.float32)
        self.py  = rng.uniform(-WY, WY, n).astype(np.float32)
        self.pz  = rng.uniform(0.0, WZ, n).astype(np.float32)
        self.pvx = np.zeros(n, np.float32)
        self.pvy = np.zeros(n, np.float32)
        self.pvz = np.zeros(n, np.float32)
        self.hue  = rng.uniform(0, 1, n).astype(np.float32)
        self.bri  = rng.uniform(0.5, 1.0, n).astype(np.float32)
        self.age  = rng.integers(0, MAX_AGE, n).astype(np.int32)
        self.scale = rng.choice([1.0, 1.0, 1.0, 2.0, 2.0, 3.0], n
                                ).astype(np.float32)
        self._rng = np.random.default_rng()

    def upload_to(self, vbo: moderngl.Buffer) -> None:
        data = np.stack([self.px, self.py, self.pz,
                         self.hue, self.bri, self.scale], axis=1
                        ).astype(np.float32)
        vbo.write(data.tobytes())

    def update(self, field: FlowField3D,
               fid_world: List[Tuple[float, float, float, float]],
               pulse: float, speed_scale: float = 1.0) -> None:
        # Bilinear XY field sample
        gx  = np.clip((self.px + WX) / (2*WX) * (field.W - 1), 0, field.W - 1.001)
        gy  = np.clip((self.py + WY) / (2*WY) * (field.H - 1), 0, field.H - 1.001)
        gxi = gx.astype(np.int32);  tx = (gx - gxi).astype(np.float32)
        gyi = gy.astype(np.int32);  ty = (gy - gyi).astype(np.float32)
        gxi1 = np.minimum(gxi + 1, field.W - 1)
        gyi1 = np.minimum(gyi + 1, field.H - 1)
        accel = FIELD_GAIN * speed_scale * (1.0 + pulse * 0.4)
        for dim, pv in enumerate([self.pvx, self.pvy]):
            f00 = field.field[gyi,  gxi,  dim]
            f10 = field.field[gyi,  gxi1, dim]
            f01 = field.field[gyi1, gxi,  dim]
            f11 = field.field[gyi1, gxi1, dim]
            s   = (f00*(1-tx)*(1-ty) + f10*tx*(1-ty) +
                   f01*(1-tx)*ty     + f11*tx*ty)
            if dim == 0:
                self.pvx = self.pvx * DRAG + s * accel
            else:
                self.pvy = self.pvy * DRAG + s * accel

        # Analytic Z
        vz = field.z_velocity(self.px, self.py, self.pz)
        self.pvz = self.pvz * DRAG_Z + vz * FIELD_GAIN * 0.15

        # Speed clamp (3D)
        speed = np.sqrt(self.pvx**2 + self.pvy**2 + (self.pvz*4)**2) + 1e-8
        cap   = PARTICLE_SPD * (1.0 + pulse * 0.25)
        over  = speed > cap
        self.pvx[over] = self.pvx[over] / speed[over] * cap
        self.pvy[over] = self.pvy[over] / speed[over] * cap
        self.pvz[over] = self.pvz[over] / speed[over] * cap * 0.25

        self.px  += self.pvx
        self.py  += self.pvy
        self.pz  += self.pvz
        self.age += 1

        # Hue drift toward nearest fiducial (XY only)
        if fid_world:
            fpos  = np.array([[wx, wy] for wx, wy, wz, fh in fid_world], np.float32)
            fhues = np.array([fh for _, _, _, fh in fid_world], np.float32)
            diff  = np.stack([self.px, self.py], 1)[:, None, :] - fpos[None, :, :]
            near  = np.argmin(diff[:, :, 0]**2 + diff[:, :, 1]**2, axis=1)
            delta = (fhues[near] - self.hue + 0.5) % 1.0 - 0.5
            self.hue = (self.hue + delta * 0.04) % 1.0

        spd2d = np.hypot(self.pvx, self.pvy)
        self.bri = np.clip(0.55 + spd2d / PARTICLE_SPD * 0.45 + pulse * 0.15,
                           0.2, 1.0)

        # Respawn
        dead = ((self.px < -WX) | (self.px > WX) |
                (self.py < -WY) | (self.py > WY) |
                (self.pz <  0)  | (self.pz > WZ) |
                (self.age > MAX_AGE))
        nd = int(dead.sum())
        if nd:
            if fid_world:
                n_near = int(nd * 0.70)
                n_rand = nd - n_near
                fpos3  = np.array([[wx, wy, wz] for wx, wy, wz, _ in fid_world],
                                  np.float32)
                ch = self._rng.integers(0, len(fpos3), n_near)
                nx = np.clip(fpos3[ch, 0] + self._rng.normal(0, WX*0.18, n_near),
                             -WX, WX)
                ny = np.clip(fpos3[ch, 1] + self._rng.normal(0, WY*0.18, n_near),
                             -WY, WY)
                nz = np.clip(fpos3[ch, 2] + self._rng.normal(0, WZ*0.12, n_near),
                             0, WZ)
                di = np.where(dead)[0]
                self.px[di[:n_near]] = nx.astype(np.float32)
                self.py[di[:n_near]] = ny.astype(np.float32)
                self.pz[di[:n_near]] = nz.astype(np.float32)
                if n_rand:
                    ri = di[n_near:]
                    self.px[ri] = self._rng.uniform(-WX, WX, n_rand).astype(np.float32)
                    self.py[ri] = self._rng.uniform(-WY, WY, n_rand).astype(np.float32)
                    self.pz[ri] = self._rng.uniform(0, WZ, n_rand).astype(np.float32)
            else:
                self.px[dead] = self._rng.uniform(-WX, WX, nd).astype(np.float32)
                self.py[dead] = self._rng.uniform(-WY, WY, nd).astype(np.float32)
                self.pz[dead] = self._rng.uniform(0, WZ, nd).astype(np.float32)
            self.pvx[dead] = self.pvy[dead] = self.pvz[dead] = 0
            self.age[dead] = 0
            self.scale[dead] = self._rng.choice(
                [1.0, 1.0, 1.0, 2.0, 2.0, 3.0], nd).astype(np.float32)


# ── Renderer ───────────────────────────────────────────────────────────────────

class Renderer3D:
    """moderngl renderer. All drawing goes through GPU; exposes high-level methods
    to visualizer.py that mirror the old LuminousRenderer API."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        self._ctx = ctx
        self.W, self.H = width, height
        self._mvp = _build_mvp(width, height)

        # Set in init()
        self._progs:    dict = {}
        self._quad_vao: dict = {}
        self._flow_fbos:   list = []   # [(fbo, tex), (fbo, tex)]
        self._interf_fbos: list = []
        self._ghost_fbos:  list = []   # constellation ghost — separate canvas
        self._flow_idx   = 0
        self._interf_idx = 0
        self._ghost_idx  = 0
        self._bloom_h_fbo = self._bloom_h_tex = None
        self._bloom_v_fbo = self._bloom_v_tex = None
        self._particle_vbo: Optional[moderngl.Buffer] = None
        self._sphere_vbo:   Optional[moderngl.Buffer] = None
        self._sphere_vao:   Optional[moderngl.VertexArray] = None
        self._particle_vao: Optional[moderngl.VertexArray] = None
        self._overlay_vbo:  Optional[moderngl.Buffer] = None
        self._overlay_vao:  Optional[moderngl.VertexArray] = None
        self._bar_vbo:      Optional[moderngl.Buffer] = None
        self._bar_vao:      Optional[moderngl.VertexArray] = None
        self._energy_tex:   Optional[moderngl.Texture] = None
        self._energy_lw = width  // 8
        self._energy_lh = height // 8
        self._energy_data = np.zeros((self._energy_lh, self._energy_lw), np.float32)
        self._hud_tex: Optional[moderngl.Texture] = None
        self._hud_surf: Optional[pygame.Surface] = None
        self._font: Optional[pygame.font.Font] = None
        self._voronoi_cache: Optional[np.ndarray] = None
        self._voronoi_fids:  Optional[np.ndarray] = None
        self._voronoi_tex:   Optional[moderngl.Texture] = None
        self._image_tex:     Optional[moderngl.Texture] = None

    # ── init ──────────────────────────────────────────────────────────────────

    def init(self) -> None:
        ctx = self._ctx
        W, H = self.W, self.H
        BW, BH = W // 2, H // 2

        # Compile shaders
        progs = {
            'decay':        ctx.program(vertex_shader=_VERT_QUAD,     fragment_shader=_FRAG_DECAY),
            'particle':     ctx.program(vertex_shader=_VERT_PARTICLE,  fragment_shader=_FRAG_PARTICLE),
            'sphere':       ctx.program(vertex_shader=_VERT_SPHERE,    fragment_shader=_FRAG_SPHERE),
            'interf':       ctx.program(vertex_shader=_VERT_QUAD,      fragment_shader=_FRAG_INTERF),
            'bloom_thresh': ctx.program(vertex_shader=_VERT_QUAD,      fragment_shader=_FRAG_BLOOM_THRESH),
            'blur_h':       ctx.program(vertex_shader=_VERT_QUAD,      fragment_shader=_FRAG_BLUR_H),
            'blur_v':       ctx.program(vertex_shader=_VERT_QUAD,      fragment_shader=_FRAG_BLUR_V),
            'composite':    ctx.program(vertex_shader=_VERT_QUAD,      fragment_shader=_FRAG_COMPOSITE),
            'overlay':      ctx.program(vertex_shader=_VERT_OVERLAY,   fragment_shader=_FRAG_OVERLAY),
            'hud':          ctx.program(vertex_shader=_VERT_QUAD,      fragment_shader=_FRAG_HUD),
            'voronoi':      ctx.program(vertex_shader=_VERT_QUAD,      fragment_shader=_FRAG_VORONOI),
            'image':        ctx.program(vertex_shader=_VERT_QUAD,      fragment_shader=_FRAG_IMAGE),
            'ghost':        ctx.program(vertex_shader=_VERT_QUAD,      fragment_shader=_FRAG_GHOST),
        }
        self._progs = progs

        # Upload MVP to particle + sphere shaders
        progs['particle']['u_mvp'].write(self._mvp)
        progs['sphere']['u_mvp'].write(self._mvp)

        # Fullscreen quad VBO/VAOs
        quad = np.array([-1,-1, 1,-1, -1,1, 1,1], np.float32)
        quad_vbo = ctx.buffer(quad.tobytes())
        for name in ('decay','interf','bloom_thresh','blur_h','blur_v',
                     'composite','hud','voronoi','image','ghost'):
            self._quad_vao[name] = ctx.vertex_array(
                progs[name], [(quad_vbo, '2f', 'a_pos')])

        # Ping-pong FBOs (RGBA 16-bit float)
        def make_fbo(w, h):
            tex = ctx.texture((w, h), 4, dtype='f2')
            tex.filter = moderngl.LINEAR, moderngl.LINEAR
            fbo = ctx.framebuffer(color_attachments=[tex])
            fbo.use(); ctx.clear(0, 0, 0, 0)
            return fbo, tex

        self._flow_fbos   = [make_fbo(W, H), make_fbo(W, H)]
        self._interf_fbos = [make_fbo(W, H), make_fbo(W, H)]
        self._ghost_fbos  = [make_fbo(W, H), make_fbo(W, H)]
        self._bloom_h_fbo, self._bloom_h_tex = make_fbo(BW, BH)
        self._bloom_v_fbo, self._bloom_v_tex = make_fbo(BW, BH)

        # Particle VBO + VAO
        self._particle_vbo = ctx.buffer(reserve=N_PARTICLES * 6 * 4, dynamic=True)
        self._particle_vao = ctx.vertex_array(
            progs['particle'],
            [(self._particle_vbo, '3f 1f 1f 1f', 'a_pos', 'a_hue', 'a_bri', 'a_size')])

        # Fiducial sphere VBO + VAO (max 64 fiducials × 4 floats: x,y,z,hue)
        self._sphere_vbo = ctx.buffer(reserve=64 * 4 * 4, dynamic=True)
        self._sphere_vao = ctx.vertex_array(
            progs['sphere'],
            [(self._sphere_vbo, '3f 1f', 'a_pos', 'a_hue')])

        # Overlay VBO + VAO (ribbons/halos — 2-D NDC)
        self._overlay_vbo = ctx.buffer(reserve=60_000 * 6 * 4, dynamic=True)
        self._overlay_vao = ctx.vertex_array(
            progs['overlay'],
            [(self._overlay_vbo, '2f 4f', 'a_pos', 'a_color')])

        # Velocity bars — dedicated buffer (64 fids × 18 verts × 6 floats × 4 bytes)
        self._bar_vbo = ctx.buffer(reserve=64 * 18 * 6 * 4, dynamic=True)
        self._bar_vao = ctx.vertex_array(
            progs['overlay'],
            [(self._bar_vbo, '2f 4f', 'a_pos', 'a_color')])

        # Energy texture (R32F, 1/8 res)
        self._energy_tex = ctx.texture(
            (self._energy_lw, self._energy_lh), 1, dtype='f4')
        self._energy_tex.filter = moderngl.LINEAR, moderngl.LINEAR

        # Voronoi texture (RGB8, 1/8 res — upscaled by GPU bilinear)
        lw, lh = self._energy_lw, self._energy_lh
        self._voronoi_tex = ctx.texture((lw, lh), 3, dtype='f1')
        self._voronoi_tex.filter = moderngl.LINEAR, moderngl.LINEAR

        # HUD texture (RGBA8, full res)
        self._hud_tex  = ctx.texture((W, H), 4)
        self._hud_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        self._font     = pygame.font.SysFont("consolas", 15)

        # GL state
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        ctx.enable(moderngl.BLEND)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _use_full(self) -> None:
        self._ctx.viewport = (0, 0, self.W, self.H)

    def _use_half(self) -> None:
        BW, BH = self.W // 2, self.H // 2
        self._ctx.viewport = (0, 0, BW, BH)

    def _ping_pong(self, fbos: list, idx: int, decay: float) -> int:
        """Decay fbos[idx] texture into fbos[1-idx]. Returns new write index."""
        src_tex = fbos[idx][1]
        dst_fbo = fbos[1 - idx][0]
        self._use_full()
        dst_fbo.use()
        self._ctx.blend_func = moderngl.ONE, moderngl.ZERO   # overwrite
        src_tex.use(location=0)
        self._progs['decay']['u_trail'] = 0
        self._progs['decay']['u_decay'] = decay
        self._quad_vao['decay'].render(moderngl.TRIANGLE_STRIP)
        return 1 - idx

    def clear_flow(self) -> None:
        for fbo, _ in self._flow_fbos:
            fbo.use(); self._ctx.clear(0, 0, 0, 0)

    def clear_interf(self) -> None:
        for fbo, _ in self._interf_fbos:
            fbo.use(); self._ctx.clear(0, 0, 0, 0)

    def invalidate_voronoi(self) -> None:
        self._voronoi_cache = None
        self._voronoi_fids  = None

    # ── FLOW mode ──────────────────────────────────────────────────────────────

    def splat_particles(self, particles: ParticleSystem3D,
                        brightness_scale: float,
                        trail_decay: float = TRAIL_DECAY) -> None:
        """Decay flow trail then additively splat particles."""
        self._flow_idx = self._ping_pong(self._flow_fbos, self._flow_idx, trail_decay)
        dst_fbo = self._flow_fbos[self._flow_idx][0]
        dst_fbo.use()
        self._ctx.blend_func = moderngl.ONE, moderngl.ONE    # additive
        particles.upload_to(self._particle_vbo)
        self._progs['particle']['u_brightness'] = brightness_scale
        self._particle_vao.render(moderngl.POINTS)

    def decay_flow(self) -> None:
        self._flow_idx = self._ping_pong(self._flow_fbos, self._flow_idx, TRAIL_DECAY)

    # ── INTERFERENCE mode ──────────────────────────────────────────────────────

    def render_interference(self, fid_3d: list, t: float,
                            global_hue: float,
                            fid_energies: Optional[List[float]],
                            awake: float) -> None:
        """Decay interference trail then render spherical waves."""
        self._interf_idx = self._ping_pong(
            self._interf_fbos, self._interf_idx, INTERF_DECAY)
        dst_fbo = self._interf_fbos[self._interf_idx][0]
        dst_fbo.use()
        self._use_full()
        self._ctx.blend_func = moderngl.ONE, moderngl.ONE

        prog = self._progs['interf']
        n = min(len(fid_3d), 12)
        prog['u_time']       = t
        prog['u_global_hue'] = global_hue
        prog['u_awake']      = awake
        prog['u_n_fids']     = n
        if n > 0:
            pos_flat = []
            for i in range(n):
                wx, wy, wz, _ = fid_3d[i]
                pos_flat.extend([wx, wy, wz])
            while len(pos_flat) < 36:
                pos_flat.extend([0, 0, 0])
            prog['u_fid_pos'].write(np.array(pos_flat[:36], np.float32).tobytes())
            e_flat = [fid_energies[i] if fid_energies and i < len(fid_energies)
                      else 0.5 for i in range(12)]
            prog['u_fid_energy'].write(np.array(e_flat, np.float32).tobytes())
        self._quad_vao['interf'].render(moderngl.TRIANGLE_STRIP)

    def decay_interf(self) -> None:
        self._interf_idx = self._ping_pong(
            self._interf_fbos, self._interf_idx, INTERF_DECAY)

    # ── CONSTELLATION mode ─────────────────────────────────────────────────────

    def _render_constellation_inner(self, dst_fbo,
                                    fid_3d: list, particles: ParticleSystem3D,
                                    global_hue: float) -> None:
        """Shared rendering logic — draws voronoi + nebula into dst_fbo."""
        dst_fbo.use()
        self._use_full()

        # Voronoi background (cached, recomputed when fids move)
        lw, lh = self._energy_lw, self._energy_lh
        recompute = True
        if fid_3d:
            fpos = np.array([[wx, wy] for wx, wy, wz, fh in fid_3d], np.float32)
            if (self._voronoi_fids is not None and
                    self._voronoi_fids.shape == fpos.shape and
                    float(np.max(np.abs(fpos - self._voronoi_fids))) < 0.01):
                recompute = False
            if recompute:
                xi = np.linspace(-WX, WX, lw); yi = np.linspace(-WY, WY, lh)
                GX, GY = np.meshgrid(xi, yi)
                pix = np.stack([GX.ravel(), GY.ravel()], 1)
                diff = pix[:, None, :] - fpos[None, :, :]
                near = np.argmin(diff[:, :, 0]**2 + diff[:, :, 1]**2, 1)
                min_d = np.sqrt(np.min(diff[:, :, 0]**2 + diff[:, :, 1]**2, 1))
                fhues = np.array([fh for _, _, _, fh in fid_3d], np.float32)
                phue  = fhues[near]
                pbri  = np.clip(0.50 - min_d / (WX * 1.4), 0.04, 0.42)
                r, g, b = _hsv_rgb_vec(phue, 1.0, pbri.astype(np.float32))
                rgb = np.stack([r, g, b], 1).reshape(lh, lw, 3).astype(np.uint8)
                self._voronoi_cache = rgb
                self._voronoi_fids  = fpos.copy()
        else:
            if self._voronoi_cache is None:
                xi = np.linspace(-WX, WX, lw)
                phue = (xi / (2*WX) + global_hue) % 1.0
                pbri = np.full(lw, 0.15, np.float32)
                r, g, b = _hsv_rgb_vec(phue, 1.0, pbri)
                rgb = np.stack([r, g, b], 1)[None, :, :].repeat(lh, 0).astype(np.uint8)
                self._voronoi_cache = rgb

        if self._voronoi_cache is not None:
            raw = self._voronoi_cache.astype(np.uint8).tobytes()
            self._voronoi_tex.write(raw)
            self._ctx.blend_func = moderngl.ONE, moderngl.ONE
            self._voronoi_tex.use(location=0)
            self._progs['voronoi']['u_tex'] = 0
            self._quad_vao['voronoi'].render(moderngl.TRIANGLE_STRIP)

        # Nebula particles
        particles.upload_to(self._particle_vbo)
        self._progs['particle']['u_brightness'] = 0.9
        self._ctx.blend_func = moderngl.ONE, moderngl.ONE
        self._particle_vao.render(moderngl.POINTS)

    def render_constellation(self, fid_3d: list,
                             particles: ParticleSystem3D,
                             global_hue: float) -> None:
        self._flow_idx = self._ping_pong(self._flow_fbos, self._flow_idx, 0.88)
        self._render_constellation_inner(
            self._flow_fbos[self._flow_idx][0], fid_3d, particles, global_hue)

    def render_ghost_constellation(self, fid_3d: list,
                                   particles: ParticleSystem3D,
                                   global_hue: float) -> None:
        """Render constellation into the dedicated ghost FBOs (separate trail canvas)."""
        self._ghost_idx = self._ping_pong(self._ghost_fbos, self._ghost_idx, 0.90)
        self._render_constellation_inner(
            self._ghost_fbos[self._ghost_idx][0], fid_3d, particles, global_hue)

    def draw_ghost_constellation(self, alpha: float,
                                 scissor_px: tuple) -> None:
        """Blend the ghost constellation over the current screen within a pixel rect.

        scissor_px: (x, y, w, h) in OpenGL pixel coords (y=0 at bottom).
        """
        if alpha <= 0.005:
            return
        ctx = self._ctx
        ghost_tex = self._ghost_fbos[self._ghost_idx][1]
        ctx.screen.use()
        self._use_full()
        ctx.scissor = scissor_px
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ghost_tex.use(location=0)
        self._progs['ghost']['u_ghost'] = 0
        self._progs['ghost']['u_alpha'] = float(alpha)
        self._quad_vao['ghost'].render(moderngl.TRIANGLE_STRIP)
        ctx.scissor = None

    # ── energy map ────────────────────────────────────────────────────────────

    def update_energy_map(self, fid_world_energies: List[Tuple]) -> None:
        lw, lh = self._energy_lw, self._energy_lh
        self._energy_data[:] = 0.0
        if not fid_world_energies:
            self._energy_tex.write(self._energy_data.tobytes())
            return
        if len(fid_world_energies) > 12:
            fid_world_energies = sorted(fid_world_energies,
                                        key=lambda e: e[3], reverse=True)[:12]
        xi = np.linspace(-WX, WX, lw, dtype=np.float32)
        yi = np.linspace(-WY, WY, lh, dtype=np.float32)
        GX, GY = np.meshgrid(xi, yi)
        sig_x = WX * 0.28;  sig_y = WY * 0.28
        for wx, wy, wz, energy in fid_world_energies:
            if energy < 0.02:
                continue
            dx = (GX - wx) / sig_x
            dy = (GY - wy) / sig_y
            self._energy_data += np.exp(-0.5 * (dx*dx + dy*dy)) * float(energy)
        np.clip(self._energy_data, 0.0, 1.0, out=self._energy_data)
        self._energy_tex.write(self._energy_data.astype(np.float32).tobytes())

    # ── overlays ──────────────────────────────────────────────────────────────

    def draw_fiducial_spheres(self, fid_3d: list,
                              sub_bass: float = 0.0, beat: float = 0.0) -> None:
        """Render each fiducial as a perspective-correct lit sphere point sprite."""
        if not fid_3d:
            return
        data = np.array(
            [[wx, wy, wz, fh] for wx, wy, wz, fh in fid_3d],
            dtype=np.float32,
        ).tobytes()
        self._sphere_vbo.write(data)
        prog = self._progs['sphere']
        prog['u_sub_bass'] = sub_bass
        prog['u_beat']     = beat
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._sphere_vao.render(moderngl.POINTS, vertices=len(fid_3d))

    def draw_aurora_ribbons(self, fid_3d: list, t: float,
                            width: int, height: int) -> None:
        if len(fid_3d) < 2:
            return
        fid_screen = [(world_to_screen(wx, wy, wz, width, height), fh)
                      for wx, wy, wz, fh in fid_3d]
        try:
            if len(fid_screen) >= 3:
                from scipy.spatial import Delaunay
                pts = np.array([s for s, _ in fid_screen])
                tri = Delaunay(pts)
                seen: set = set()
                pairs = []
                for simp in tri.simplices:
                    for a, b in [(0,1),(1,2),(0,2)]:
                        e = (min(simp[a],simp[b]), max(simp[a],simp[b]))
                        if e not in seen:
                            seen.add(e); pairs.append(e)
            else:
                pairs = [(0, 1)]
        except Exception:
            pairs = [(i,j) for i in range(len(fid_screen))
                     for j in range(i+1, len(fid_screen))]

        verts = []
        segments: list = []
        N = 20
        max_verts = self._overlay_vbo.size // (6 * 4)

        for a, b in pairs:
            (sx0, sy0), fh0 = fid_screen[a]
            (sx1, sy1), fh1 = fid_screen[b]
            if math.hypot(sx1-sx0, sy1-sy0) > width * 0.85:
                continue
            start = len(verts) // 6
            if start + N > max_verts:
                break
            seed  = a * 31 + b * 17
            mid_x = (sx0+sx1)*0.5 + math.sin(t*0.7 + seed) * width * 0.055
            mid_y = (sy0+sy1)*0.5 + math.cos(t*0.5 + seed*1.3) * height * 0.063
            for ki in range(N):
                tt    = ki / (N - 1)
                bx    = (1-tt)**2*sx0 + 2*(1-tt)*tt*mid_x + tt**2*sx1
                by    = (1-tt)**2*sy0 + 2*(1-tt)*tt*mid_y + tt**2*sy1
                alpha = math.sin(tt * math.pi) * 0.55
                hue   = (fh0 * (1-tt) + fh1 * tt) % 1.0
                r, g, b_c = _hue_to_rgb_f(hue)
                nx = bx / width  * 2 - 1
                ny = 1 - by / height * 2
                verts.extend([nx, ny, r, g, b_c, alpha])
            segments.append((start, N))

        if not verts:
            return
        data = np.array(verts, np.float32).tobytes()
        self._overlay_vbo.write(data)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        for first, count in segments:
            self._overlay_vao.render(moderngl.LINE_STRIP, vertices=count, first=first)

    def draw_halos(self, fid_3d: list, t: float,
                   pulse: float, width: int, height: int,
                   sub_bass: float = 0.0, beat: float = 0.0) -> None:
        if not fid_3d:
            return
        verts = []
        segments: list = []
        SEGS = 48
        max_verts = self._overlay_vbo.size // (6 * 4)

        for wx, wy, wz, fh in fid_3d:
            sx, sy = world_to_screen(wx, wy, wz, width, height)
            depth  = wz + 1.0
            # Per-fiducial slow breathe — hue gives a unique phase so they desync
            phase_off  = fh * math.tau
            breathe    = 0.30 * math.sin(t * 0.45 + phase_off)
            local_sub  = max(0.0, sub_bass + breathe)
            base_r = (22 + local_sub * 22 + beat * 12) / depth
            r, g, b_c = _hue_to_rgb_f(fh)
            # Slowly drifting corona — gentle, lazy rotation
            for ring_i, (ring_scale, spin_dir, spin_speed) in enumerate([
                    (0.6,  1.0, 0.10),
                    (0.9, -1.0, 0.06),
                    (1.15, 1.0, 0.03),
            ]):
                rr    = base_r * ring_scale * (1.0 + beat * 0.22)
                alpha = (0.65 - ring_i * 0.15) * (0.40 + local_sub * 0.35)
                start = len(verts) // 6
                if start + SEGS + 1 > max_verts:
                    break
                for ki in range(SEGS + 1):
                    angle = 2*math.pi*ki/SEGS + t * spin_speed * spin_dir + ring_i * 1.1
                    nx = (sx + math.cos(angle) * rr) / width  * 2 - 1
                    ny = 1 - (sy + math.sin(angle) * rr) / height * 2
                    verts.extend([nx, ny, r, g, b_c, alpha])
                segments.append((start, SEGS + 1))
            # Outer glow rings
            for ring_scale in (1.6, 2.4, 3.8):
                rr    = base_r * ring_scale
                alpha = 0.28 / ring_scale * (1.0 + local_sub * 0.8)
                start = len(verts) // 6
                if start + SEGS + 1 > max_verts:
                    break
                for ki in range(SEGS + 1):
                    angle = 2*math.pi*ki/SEGS + t * 0.15
                    nx = (sx + math.cos(angle) * rr) / width  * 2 - 1
                    ny = 1 - (sy + math.sin(angle) * rr) / height * 2
                    verts.extend([nx, ny, r, g, b_c, alpha])
                segments.append((start, SEGS + 1))

        if not verts:
            return
        data = np.array(verts, np.float32).tobytes()
        self._overlay_vbo.write(data)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        for first, count in segments:
            self._overlay_vao.render(moderngl.LINE_STRIP, vertices=count, first=first)

    def draw_velocity_bars(self, fid_world: list, fid_vels: list,
                           width: int, height: int) -> None:
        """Thick axis-aligned bars extending from each fiducial in its dominant
        movement direction.  Horizontal mover → wide horizontal bar.
        Vertical mover → tall vertical bar.  Very subtle, fades with speed."""
        if not fid_world:
            return

        verts = []
        aspect = width / height

        for i, (wx, wy, wz, fh) in enumerate(fid_world):
            if i >= len(fid_vels):
                break
            evx, evy = fid_vels[i]
            speed = math.hypot(evx, evy)
            if speed < 0.0012:          # too slow — skip
                continue

            sx, sy = world_to_screen(wx, wy, wz, width, height)
            cx = sx / width  * 2 - 1
            cy = 1 - sy / height * 2

            r, g, b = _hue_to_rgb_f(fh)
            t_speed  = min(speed / 0.007, 1.0)   # 0..1 normalised speed
            alpha    = t_speed * 0.30             # max 30% opacity — subtle

            # Dominant axis determines bar orientation
            if abs(evx) >= abs(evy):
                # Horizontal bar: long in X, narrow in Y
                half_len = t_speed * 0.38          # NDC half-length
                half_w   = 5.0 / height            # NDC half-width (~5 px)
                x0, x1   = cx - half_len, cx + half_len
                y0, y1   = cy - half_w,   cy + half_w
            else:
                # Vertical bar: long in Y, narrow in X
                half_len = t_speed * 0.38
                half_w   = 5.0 / width
                x0, x1   = cx - half_w,   cx + half_w
                y0, y1   = cy - half_len,  cy + half_len

            # Two triangles — centre alpha, ends fade to 0
            # Centre strip (full alpha) and outer ends (alpha 0)
            if abs(evx) >= abs(evy):
                cx0, cx1 = cx - half_w * 3, cx + half_w * 3   # narrow centre band
                # Left fade quad
                verts += [x0,  y0, r, g, b, 0.0,
                          x0,  y1, r, g, b, 0.0,
                          cx0, y0, r, g, b, alpha,
                          x0,  y1, r, g, b, 0.0,
                          cx0, y1, r, g, b, alpha,
                          cx0, y0, r, g, b, alpha]
                # Centre quad (solid)
                verts += [cx0, y0, r, g, b, alpha,
                          cx0, y1, r, g, b, alpha,
                          cx1, y0, r, g, b, alpha,
                          cx0, y1, r, g, b, alpha,
                          cx1, y1, r, g, b, alpha,
                          cx1, y0, r, g, b, alpha]
                # Right fade quad
                verts += [cx1, y0, r, g, b, alpha,
                          cx1, y1, r, g, b, alpha,
                          x1,  y0, r, g, b, 0.0,
                          cx1, y1, r, g, b, alpha,
                          x1,  y1, r, g, b, 0.0,
                          x1,  y0, r, g, b, 0.0]
            else:
                cy0, cy1 = cy - half_w * 3, cy + half_w * 3
                # Bottom fade
                verts += [x0, y0,  r, g, b, 0.0,
                          x1, y0,  r, g, b, 0.0,
                          x0, cy0, r, g, b, alpha,
                          x1, y0,  r, g, b, 0.0,
                          x1, cy0, r, g, b, alpha,
                          x0, cy0, r, g, b, alpha]
                # Centre
                verts += [x0, cy0, r, g, b, alpha,
                          x1, cy0, r, g, b, alpha,
                          x0, cy1, r, g, b, alpha,
                          x1, cy0, r, g, b, alpha,
                          x1, cy1, r, g, b, alpha,
                          x0, cy1, r, g, b, alpha]
                # Top fade
                verts += [x0, cy1, r, g, b, alpha,
                          x1, cy1, r, g, b, alpha,
                          x0, y1,  r, g, b, 0.0,
                          x1, cy1, r, g, b, alpha,
                          x1, y1,  r, g, b, 0.0,
                          x0, y1,  r, g, b, 0.0]

        if not verts:
            return
        data = np.array(verts, np.float32).tobytes()
        self._bar_vbo.write(data)
        self._ctx.screen.use()
        self._use_full()
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self._bar_vao.render(moderngl.TRIANGLES, vertices=len(verts) // 6)

    def draw_wave_rings(self, ring_events: list, t: float,
                        width: int, height: int) -> None:
        """Expanding ring waves from fiducial positions on beat.
        ring_events: list of (birth_t, cx_ndc, cy_ndc, r, g, b)
        """
        if not ring_events:
            return
        SEGS = 48
        LIFETIME = 1.8
        aspect = width / height
        max_verts = self._overlay_vbo.size // (6 * 4)
        verts = []
        segments: list = []   # (first, count) per ring to avoid cross-ring artifacts

        for birth_t, cx, cy, r, g, b in ring_events:
            age = t - birth_t
            if age >= LIFETIME:
                continue
            frac = age / LIFETIME
            alpha = (1.0 - frac) ** 1.5 * 0.9
            radius = frac * 1.1
            for ring_i in range(2):
                r_off = radius + ring_i * 0.025
                a_off = alpha * (1.0 - ring_i * 0.5)
                start = len(verts) // 6
                if start + SEGS + 1 > max_verts:
                    break
                for ki in range(SEGS + 1):
                    angle = 2 * math.pi * ki / SEGS
                    px = cx + math.cos(angle) * r_off
                    py = cy + math.sin(angle) * r_off * aspect
                    verts.extend([px, py, r, g, b, a_off])
                segments.append((start, SEGS + 1))

        if not verts:
            return
        data = np.array(verts, np.float32).tobytes()
        self._overlay_vbo.write(data)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        # Draw each ring separately so rings don't connect to each other
        for first, count in segments:
            self._overlay_vao.render(moderngl.LINE_STRIP, vertices=count, first=first)

    # ── composite + bloom ─────────────────────────────────────────────────────

    def composite(self, blend_weight: float, bloom_str: float,
                  constellation: bool = False) -> None:
        ctx = self._ctx
        W, H = self.W, self.H
        BW, BH = W // 2, H // 2

        flow_tex   = self._flow_fbos[self._flow_idx][1]
        interf_tex = self._interf_fbos[self._interf_idx][1]

        # Bloom threshold at half res
        self._bloom_h_fbo.use()
        ctx.viewport = (0, 0, BW, BH)
        ctx.blend_func = moderngl.ONE, moderngl.ZERO
        flow_tex.use(location=0)
        self._progs['bloom_thresh']['u_scene']     = 0
        self._progs['bloom_thresh']['u_threshold'] = 0.55
        self._quad_vao['bloom_thresh'].render(moderngl.TRIANGLE_STRIP)

        # Blur H
        self._bloom_v_fbo.use()
        self._bloom_h_tex.use(location=0)
        self._progs['blur_h']['u_tex']   = 0
        self._progs['blur_h']['u_texel'] = (1.0/BW, 0.0)
        self._quad_vao['blur_h'].render(moderngl.TRIANGLE_STRIP)

        # Blur V
        self._bloom_h_fbo.use()
        self._bloom_v_tex.use(location=0)
        self._progs['blur_v']['u_tex']   = 0
        self._progs['blur_v']['u_texel'] = (0.0, 1.0/BH)
        self._quad_vao['blur_v'].render(moderngl.TRIANGLE_STRIP)

        # Final composite → screen
        ctx.screen.use()
        ctx.viewport = (0, 0, W, H)
        ctx.blend_func = moderngl.ONE, moderngl.ZERO

        prog = self._progs['composite']
        flow_tex.use(location=0);         prog['u_flow']      = 0
        interf_tex.use(location=1);       prog['u_interf']    = 1
        self._bloom_h_tex.use(location=2); prog['u_bloom']    = 2
        self._energy_tex.use(location=3);  prog['u_energy']   = 3
        prog['u_blend']     = 0.0 if constellation else blend_weight
        prog['u_bloom_str'] = bloom_str
        self._quad_vao['composite'].render(moderngl.TRIANGLE_STRIP)

    # ── HUD ───────────────────────────────────────────────────────────────────

    def draw_image_overlay(self, rgba_bytes: bytes, img_w: int, img_h: int,
                           alpha: float, global_hue: float) -> None:
        """Blit a full-screen image at very low opacity. rgba_bytes = RGBA uint8."""
        if alpha <= 0.001:
            return
        ctx = self._ctx
        if (self._image_tex is None
                or self._image_tex.width != img_w
                or self._image_tex.height != img_h):
            if self._image_tex:
                self._image_tex.release()
            self._image_tex = ctx.texture((img_w, img_h), 4)
            self._image_tex.filter = moderngl.LINEAR, moderngl.LINEAR
        self._image_tex.write(rgba_bytes)
        ctx.screen.use()
        self._use_full()
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._image_tex.use(location=0)
        prog = self._progs['image']
        prog['u_tex']        = 0
        prog['u_alpha']      = float(alpha)
        prog['u_global_hue'] = float(global_hue)
        self._quad_vao['image'].render(moderngl.TRIANGLE_STRIP)

    def draw_hud_surface(self, lines: List[str],
                          fid_labels: Optional[List] = None) -> None:
        """Render HUD text lines + optional per-fiducial coordinate labels.

        fid_labels: list of (sx, sy, x_mm, y_mm, z_mm, r, g, b) tuples.
        """
        if self._font is None or self._hud_surf is None:
            return
        self._hud_surf.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            surf = self._font.render(line, True, (90, 90, 90))
            self._hud_surf.blit(surf, (12, 10 + i * 20))
        if fid_labels:
            for sx, sy, xm, ym, zm, r, g, b in fid_labels:
                col   = (int(r * 200 + 40), int(g * 200 + 40), int(b * 200 + 40))
                text  = f"{xm:+.0f} {ym:+.0f} {zm:.0f}"
                label = self._font.render(text, True, col)
                # Offset label slightly up-right from the fiducial dot
                self._hud_surf.blit(label, (int(sx) + 12, int(sy) - 10))
        raw = pygame.image.tostring(self._hud_surf, 'RGBA', True)
        self._hud_tex.write(raw)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._hud_tex.use(location=0)
        self._progs['hud']['u_tex'] = 0
        self._quad_vao['hud'].render(moderngl.TRIANGLE_STRIP)
