import os
import pygame
import math
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 1024, 768
LEFT_PANEL_WIDTH = 600
RIGHT_PANEL_WIDTH = WINDOW_WIDTH - LEFT_PANEL_WIDTH
FPS = 60
G = 9.8  # Gravity (m/s²)
THETA = math.radians(30)  # 30° incline
SIN_THETA, COS_THETA = math.sin(THETA), math.cos(THETA)
A = G * SIN_THETA  # Acceleration along incline (4.9 m/s²)
L_PHYSICAL_MAX = 2.5  # Max plane length (m)
L_PIXEL_MAX = 400  # Max plane length in pixels

# Experimental data
HEIGHTS = [0.25, 0.50, 0.75, 1.00, 1.25]  # meters
PLANE_LENGTHS = [0.50, 1.00, 1.50, 2.00, 2.50]  # meters
EXP_AVG_SPEEDS = [1.11, 1.0, 1.5, 2.0, 2.5]  # m/s
EXP_STD_DEVS = [0.03, 0.04, 0.03, 0.03, 0.04]  # m/s

# Colors
WHITE = (255, 255, 255)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
RED = (200, 0, 0)
NAVY = (0, 0, 100)
BLACK = (0, 0, 0)
LIGHT_BLUE = (200, 200, 255)
GREEN = (0, 200, 0)
YELLOW = (255, 255, 0)

class Button:
    def __init__(self, x, y, width, height, text, color=NAVY, hover_color=LIGHT_BLUE, text_color=WHITE):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = pygame.font.SysFont('Arial', 18)
        self.clicked = False

    def draw(self, surface, is_hovered):
        color = self.hover_color if is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, BLACK, self.rect, 2, border_radius=5)
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def is_hovered(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)

    def is_clicked(self, event, mouse_pos):
        if event.type == pygame.MOUSEBUTTONDOWN and self.is_hovered(mouse_pos):
            self.clicked = True
            return True
        return False

class GraphPlot:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.points = []  # (s, v) pairs
        self.font = pygame.font.SysFont('Arial', 14)

    def add_point(self, s, v):
        self.points.append((s, v))

    def draw(self, surface, selected_height):
        pygame.draw.rect(surface, WHITE, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 1)

        x_axis_y = self.rect.bottom - 20
        y_axis_x = self.rect.left + 20
        pygame.draw.line(surface, BLACK, (self.rect.left, x_axis_y), (self.rect.right, x_axis_y), 2)
        pygame.draw.line(surface, BLACK, (y_axis_x, self.rect.top), (y_axis_x, self.rect.bottom), 2)

        surface.blit(self.font.render("s (m)", True, BLACK), (self.rect.right - 30, x_axis_y + 5))
        surface.blit(self.font.render("v (m/s)", True, BLACK), (y_axis_x - 15, self.rect.top - 15))

        max_s = PLANE_LENGTHS[-1]
        max_v = 4.0
        for s, v in self.points:
            if s > max_s:
                continue
            x = self.rect.left + 20 + (s / max_s) * (self.rect.width - 40)
            y = self.rect.bottom - 20 - (v / max_v) * (self.rect.height - 40)
            pygame.draw.circle(surface, RED, (int(x), int(y)), 2)

        if selected_height in HEIGHTS:
            idx = HEIGHTS.index(selected_height)
            L = PLANE_LENGTHS[idx]
            h_array = np.array(HEIGHTS)
            v_exp = np.array(EXP_AVG_SPEEDS)
            slope, intercept, r_value, _, _ = stats.linregress(h_array, v_exp)
            x_vals = np.linspace(0, max_s, 100)
            v_linear = slope * (x_vals / 2) + intercept
            for i in range(len(x_vals) - 1):
                x1 = self.rect.left + 20 + (x_vals[i] / max_s) * (self.rect.width - 40)
                y1 = self.rect.bottom - 20 - (v_linear[i] / max_v) * (self.rect.height - 40)
                x2 = self.rect.left + 20 + (x_vals[i + 1] / max_s) * (self.rect.width - 40)
                y2 = self.rect.bottom - 20 - (v_linear[i + 1] / max_v) * (self.rect.height - 40)
                pygame.draw.line(surface, GREEN, (x1, y1), (x2, y2), 1)

            def power_law(h, A_fit, p):
                return A_fit * h ** p
            popt, _ = curve_fit(power_law, h_array, v_exp)
            A_fit, p = popt
            v_power = power_law(x_vals / 2, A_fit, p)
            for i in range(len(x_vals) - 1):
                x1 = self.rect.left + 20 + (x_vals[i] / max_s) * (self.rect.width - 40)
                y1 = self.rect.bottom - 20 - (v_power[i] / max_v) * (self.rect.height - 40)
                x2 = self.rect.left + 20 + (x_vals[i + 1] / max_s) * (self.rect.width - 40)
                y2 = self.rect.bottom - 20 - (v_power[i + 1] / max_v) * (self.rect.height - 40)
                pygame.draw.line(surface, YELLOW, (x1, y1), (x2, y2), 1)

            s_highlight = L
            v_highlight = v_exp[idx]
            x_h = self.rect.left + 20 + (s_highlight / max_s) * (self.rect.width - 40)
            y_h = self.rect.bottom - 20 - (v_highlight / max_v) * (self.rect.height - 40)
            pygame.draw.circle(surface, BLACK, (int(x_h), int(y_h)), 5)

    def clear(self):
        self.points = []

class InclineScene:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.incline_start = (self.rect.left + 50, self.rect.bottom - 50)  # Adjusted starting point
        self.block_size = 30
        self.block_pos = self.incline_start
        self.t = 0
        self.s = 0
        self.L = PLANE_LENGTHS[0]

    def set_height(self, h):
        idx = HEIGHTS.index(h)
        self.L = PLANE_LENGTHS[idx]
        self.reset()

    def reset(self):
        self.block_pos = self.incline_start
        self.t = 0
        self.s = 0

    def update(self, dt):
        if self.s < self.L:
            self.t += dt
            self.s = 0.5 * A * self.t ** 2
            if self.s > self.L:
                self.s = self.L
            pixel_s = (self.s / L_PHYSICAL_MAX) * L_PIXEL_MAX
            delta_x = pixel_s * COS_THETA
            delta_y = pixel_s * SIN_THETA
            self.block_pos = (self.incline_start[0] + delta_x, self.incline_start[1] - delta_y)  # Adjusted y-axis
        return self.s, A * self.t

    def draw(self, surface):
        incline_end = (self.incline_start[0] + L_PIXEL_MAX * COS_THETA,
                       self.incline_start[1] - L_PIXEL_MAX * SIN_THETA)  # Adjusted y-axis
        pygame.draw.line(surface, DARK_GRAY, self.incline_start, incline_end, 5)
        
        for i in range(0, int(L_PHYSICAL_MAX * 2) + 1):
            s = i * 0.5
            pixel_s = (s / L_PHYSICAL_MAX) * L_PIXEL_MAX
            x = self.incline_start[0] + pixel_s * COS_THETA
            y = self.incline_start[1] - pixel_s * SIN_THETA  # Adjusted y-axis
            pygame.draw.circle(surface, BLACK, (int(x), int(y)), 3)

        block_rect = pygame.Rect(self.block_pos[0] - self.block_size // 2,
                                 self.block_pos[1] - self.block_size // 2,
                                 self.block_size, self.block_size)
        pygame.draw.rect(surface, RED, block_rect)

class SimulationController:
    def __init__(self):
        self.selected_height = HEIGHTS[0]
        self.running = False
        self.paused = False
        self.scene = InclineScene(0, 100, LEFT_PANEL_WIDTH, WINDOW_HEIGHT - 100)
        self.graph = GraphPlot(LEFT_PANEL_WIDTH + 20, 400, RIGHT_PANEL_WIDTH - 40, 200)
        self.buttons = [
            Button(LEFT_PANEL_WIDTH + 20, 150 + i * 40, 150, 30, f"h = {h} m")
            for i, h in enumerate(HEIGHTS)
        ]
        self.start_button = Button(LEFT_PANEL_WIDTH + 20, 350, 150, 30, "Start")
        self.reset_button = Button(LEFT_PANEL_WIDTH + 20, 390, 150, 30, "Reset")
        self.font = pygame.font.SysFont('Arial', 18)
        self.header_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.fit_results = None
        self.match_status = ""

    def handle_events(self, events, mouse_pos):
        for event in events:
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
            for btn in self.buttons:
                if btn.is_clicked(event, mouse_pos):
                    self.selected_height = float(btn.text.split('=')[1].strip().split()[0])
                    self.scene.set_height(self.selected_height)
                    self.running = False
                    self.paused = False
                    self.start_button.text = "Start"
                    self.graph.clear()
                    self.fit_results = None
                    self.match_status = ""
            if self.start_button.is_clicked(event, mouse_pos):
                if not self.running:
                    self.running = True
                    self.paused = False
                    self.start_button.text = "Pause"
                else:
                    self.paused = not self.paused
                    self.start_button.text = "Resume" if self.paused else "Pause"
            if self.reset_button.is_clicked(event, mouse_pos):
                self.scene.reset()
                self.running = False
                self.paused = False
                self.start_button.text = "Start"
                self.graph.clear()
                self.fit_results = None
                self.match_status = ""
        return True

    def update(self, dt):
        if self.running and not self.paused:
            s, v = self.scene.update(dt)
            self.graph.add_point(s, v)
            if s >= self.scene.L:
                self.running = False
                self.start_button.text = "Start"
                idx = HEIGHTS.index(self.selected_height)
                actual_time = self.scene.t
                measured_v = self.scene.L / actual_time if actual_time > 0 else 0
                exp_v = EXP_AVG_SPEEDS[idx]
                exp_sd = EXP_STD_DEVS[idx]
                self.match_status = "Success" if abs(measured_v - exp_v) <= exp_sd else "Deviation"
                h_array = np.array(HEIGHTS)
                v_exp = np.array(EXP_AVG_SPEEDS)
                slope, intercept, r_value, _, _ = stats.linregress(h_array, v_exp)
                def power_law(h, A_fit, p):
                    return A_fit * h ** p
                popt, _ = curve_fit(power_law, h_array, v_exp)
                A_fit, p = popt
                self.fit_results = {
                    'linear': (slope, intercept, r_value ** 2),
                    'power': (A_fit, p)
                }

    def draw(self, surface):
        surface.fill(LIGHT_GRAY)
        pygame.draw.rect(surface, LIGHT_BLUE, (LEFT_PANEL_WIDTH, 0, RIGHT_PANEL_WIDTH, WINDOW_HEIGHT))
        pygame.draw.rect(surface, BLACK, (50, 50, 100, 100))  # Debug rectangle
        
        header = self.header_font.render("Inclined-Plane Velocity Simulation", True, BLACK)
        surface.blit(header, (WINDOW_WIDTH // 2 - header.get_width() // 2, 20))

        self.scene.draw(surface)

        mouse_pos = pygame.mouse.get_pos()
        for btn in self.buttons:
            is_hovered = btn.is_hovered(mouse_pos)
            btn.draw(surface, is_hovered)
        self.start_button.draw(surface, self.start_button.is_hovered(mouse_pos))
        self.reset_button.draw(surface, self.reset_button.is_hovered(mouse_pos))

        idx = HEIGHTS.index(self.selected_height)
        L = PLANE_LENGTHS[idx]
        t_theory = math.sqrt(2 * L / A)
        v_theory = L / t_theory
        data_texts = [
            f"Height: {self.selected_height:.2f} m",
            f"Length: {L:.2f} m",
            f"Exp. Avg. Speed: {EXP_AVG_SPEEDS[idx]:.2f} m/s",
            f"Exp. SD: {EXP_STD_DEVS[idx]:.2f} m/s",
            f"Theor. Avg. Speed: {v_theory:.2f} m/s",
            f"Theor. Time: {t_theory:.2f} s",
            f"Elapsed Time: {self.scene.t:.2f} s",
            f"Inst. Speed: {A * self.scene.t:.2f} m/s"
        ]
        if self.match_status:
            data_texts.append(f"Match: {self.match_status}")
        if self.fit_results:
            slope, intercept, r2 = self.fit_results['linear']
            A_fit, p = self.fit_results['power']
            data_texts.extend([
                f"Linear Fit: v = {slope:.2f}h + {intercept:.2f}",
                f"Linear R²: {r2:.4f}",
                f"Power Fit: v = {A_fit:.2f}h^{p:.2f}"
            ])
        for i, text in enumerate(data_texts):
            surface.blit(self.font.render(text, True, BLACK), (LEFT_PANEL_WIDTH + 20, 600 + i * 20))

        self.graph.draw(surface, self.selected_height)

        fps = str(int(clock.get_fps()))
        surface.blit(self.font.render(f"FPS: {fps}", True, BLACK), (10, 10))

def main():
    try:
        print("Initializing Pygame...")
        os.environ["PYSDL2_SURFACE"] = "pygame-canvas"
        pygame.init()
        print("Pygame initialized, setting mode...")
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Inclined Plane Simulation")
        global clock
        clock = pygame.time.Clock()
        print("Clock and controller setup...")
        controller = SimulationController()

        running = True
        while running:
            events = pygame.event.get()
            mouse_pos = pygame.mouse.get_pos()
            running = controller.handle_events(events, mouse_pos)
            controller.update(1.0 / FPS)
            controller.draw(screen)
            pygame.display.flip()
            clock.tick(FPS)
        print("Simulation loop exited.")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
