import pygame
import sys
import threading
import time
from dots_and_boxes import DotsAndBoxes
from mlp_evaluator import MLPEvaluator
from minmax import MinMaxAgent
from adaptive_strategy import get_all_strategies

# Constants
# Constants
WINDOW_WIDTH = 1000 # Slightly reduced width
WINDOW_HEIGHT = 600
DOT_RADIUS = 8      # Reduced from 10
DOT_SPACING = 80    # Reduced from 100
LINE_WIDTH = 4      # Reduced from 5
HITBOX_WIDTH = 30   # Reduced from 40
DEBUG_HITBOXES = False # Set to True to see hitboxes

# Theme & Colors
class Theme:
    BG = (240, 242, 245)      # Light Gray/Blue
    PRIMARY = (44, 62, 80)    # Dark Slate
    SECONDARY = (52, 73, 94)  # Lighter Slate
    ACCENT = (52, 152, 219)   # Bright Blue
    TEXT = (44, 62, 80)       # Dark Slate
    WHITE = (255, 255, 255)
    RED = (231, 76, 60)       # Flat Red
    BLUE = (52, 152, 219)     # Flat Blue
    GREEN = (46, 204, 113)    # Flat Green
    GRAY = (149, 165, 166)    # Flat Gray
    SHADOW = (200, 200, 200)

    @staticmethod
    def get_font(size, bold=False):
        return pygame.font.SysFont("Arial", size, bold=bold)

# Global Colors mapped to Theme
WHITE = Theme.WHITE
BLACK = Theme.TEXT
RED = Theme.RED
BLUE = Theme.BLUE
GREEN = Theme.GREEN
GRAY = Theme.GRAY
LIGHT_GRAY = Theme.BG
DARK_GRAY = Theme.SECONDARY

class Button:
    def __init__(self, x, y, width, height, text, action=None, color=Theme.GRAY, hover_color=Theme.SECONDARY):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.color = color
        self.hover_color = hover_color
        self.font = Theme.get_font(24, bold=True) # Reduced from 28

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = self.rect.collidepoint(mouse_pos)
        color = self.hover_color if is_hovered else self.color
        
        # Shadow
        shadow_rect = self.rect.copy()
        shadow_rect.y += 4
        pygame.draw.rect(screen, Theme.SHADOW, shadow_rect, border_radius=8)
        
        # Button
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        
        # Text
        text_color = Theme.WHITE if color != Theme.GRAY else Theme.TEXT
        text_surf = self.font.render(self.text, True, text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos) and self.action:
                self.action()

class Selector:
    def __init__(self, x, y, width, height, options, selected_index=0, label=""):
        self.rect = pygame.Rect(x, y, width, height)
        self.options = options
        self.selected_index = selected_index
        self.label = label
        self.font = Theme.get_font(24) # Reduced from 28
        self.label_font = Theme.get_font(20, bold=True) # Reduced from 24
        
        # Buttons
        self.left_rect = pygame.Rect(x, y, 40, height)
        self.right_rect = pygame.Rect(x + width - 40, y, 40, height)

    def draw(self, screen):
        # Draw label
        if self.label:
            label_surf = self.label_font.render(self.label, True, Theme.SECONDARY)
            screen.blit(label_surf, (self.rect.x, self.rect.y - 25))

        # Shadow
        shadow_rect = self.rect.copy()
        shadow_rect.y += 2
        pygame.draw.rect(screen, Theme.SHADOW, shadow_rect, border_radius=5)

        # Draw main box background
        pygame.draw.rect(screen, Theme.WHITE, self.rect, border_radius=5)
        pygame.draw.rect(screen, Theme.GRAY, self.rect, 1, border_radius=5)
        
        # Draw arrows
        # Left arrow
        pygame.draw.rect(screen, Theme.BG, self.left_rect, border_top_left_radius=5, border_bottom_left_radius=5)
        pygame.draw.polygon(screen, Theme.SECONDARY, [
            (self.left_rect.centerx + 5, self.left_rect.centery - 8),
            (self.left_rect.centerx - 5, self.left_rect.centery),
            (self.left_rect.centerx + 5, self.left_rect.centery + 8)
        ])
        
        # Right arrow
        pygame.draw.rect(screen, Theme.BG, self.right_rect, border_top_right_radius=5, border_bottom_right_radius=5)
        pygame.draw.polygon(screen, Theme.SECONDARY, [
            (self.right_rect.centerx - 5, self.right_rect.centery - 8),
            (self.right_rect.centerx + 5, self.right_rect.centery),
            (self.right_rect.centerx - 5, self.right_rect.centery + 8)
        ])
        
        # Draw text
        text = self.options[self.selected_index] if self.options else ""
        text_surf = self.font.render(text, True, Theme.TEXT)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.left_rect.collidepoint(event.pos):
                self.selected_index = (self.selected_index - 1) % len(self.options)
                return True
            elif self.right_rect.collidepoint(event.pos):
                self.selected_index = (self.selected_index + 1) % len(self.options)
                return True
        return False
        return False

class ScrollableGameLog:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.messages = []
        self.font = pygame.font.Font(None, 24)
        self.scroll_offset = 0
        self.visible_lines = (height - 50) // 20
        self.total_height = 0

    def add_message(self, message):
        self.messages.append(message)
        self.total_height = len(self.messages) * 20
        # Auto-scroll to bottom
        if len(self.messages) > self.visible_lines:
            self.scroll_offset = len(self.messages) - self.visible_lines

    def handle_scroll(self, event):
        if event.type == pygame.MOUSEWHEEL:
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                self.scroll_offset -= event.y
                # Clamp scroll
                max_scroll = max(0, len(self.messages) - self.visible_lines)
                self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))

    def draw(self, screen):
        # Shadow
        shadow_rect = self.rect.copy()
        shadow_rect.y += 2
        pygame.draw.rect(screen, Theme.SHADOW, shadow_rect, border_radius=5)

        # Draw background
        pygame.draw.rect(screen, Theme.WHITE, self.rect, border_radius=5)
        pygame.draw.rect(screen, Theme.GRAY, self.rect, 1, border_radius=5)
        
        # Draw title
        title_surf = self.font.render("Log Partita", True, Theme.PRIMARY)
        screen.blit(title_surf, (self.rect.x + 10, self.rect.y + 10))
        pygame.draw.line(screen, Theme.GRAY, (self.rect.x, self.rect.y + 35), (self.rect.right, self.rect.y + 35), 1)

        # Draw messages with clipping
        clip_rect = pygame.Rect(self.rect.x + 5, self.rect.y + 40, self.rect.width - 10, self.rect.height - 45)
        screen.set_clip(clip_rect)
        
        start_y = self.rect.y + 45
        for i, msg in enumerate(self.messages):
            if i >= self.scroll_offset:
                y_pos = start_y + (i - self.scroll_offset) * 20
                if y_pos > self.rect.bottom:
                    break
                text_surf = self.font.render(msg, True, BLACK)
                screen.blit(text_surf, (self.rect.x + 10, y_pos))
        
        screen.set_clip(None)
        
        # Draw scrollbar if needed
        if len(self.messages) > self.visible_lines:
            scrollbar_height = (self.visible_lines / len(self.messages)) * (self.rect.height - 40)
            scrollbar_y = self.rect.y + 40 + (self.scroll_offset / len(self.messages)) * (self.rect.height - 40)
            pygame.draw.rect(screen, DARK_GRAY, (self.rect.right - 10, scrollbar_y, 5, scrollbar_height))
class MainMenu:
    def __init__(self, app):
        self.app = app
        self.strategies = list(get_all_strategies().keys())
        
        # UI Elements - Left Side (Controls)
        start_x = 50
        self.grid_size_sel = Selector(start_x, 150, 300, 50, ["2x2", "3x3", "4x4"], 1, "Grid Size")
        self.p1_type_sel = Selector(start_x, 250, 300, 50, ["Human", "AI"], 0, "Player 1 (Red)")
        self.p2_type_sel = Selector(start_x, 350, 300, 50, ["Human", "AI"], 1, "Player 2 (Blue)")
        self.strategy_sel = Selector(start_x, 450, 300, 50, self.strategies, 0, "AI Strategy") # Default to Progressive
        
        self.start_btn = Button(start_x, 550, 140, 60, "Gioca", self.start_game, Theme.GREEN, (40, 180, 100))
        # Benchmark button removed as requested
        
        # Strategy descriptions (Italian with Theory from HW1 PDF)
        self.strategy_descriptions = {
            'progressive': "STRATEGIA: Progressive Deepening [L(t), K(t)]\n\nL'agente inizia con una ricerca superficiale e aumenta la profondità L(t) man mano che la partita avanza. Questo permette di vedere più lontano quando le mosse possibili diminuiscono.\n\nL(t): 1 -> Max (Crescente)\nK(t): Costante",
            'inverse': "STRATEGIA: Relazione Inversa [L(t), K(t)]\n\nAll'aumentare della profondità di ricerca L(t), l'agente riduce il numero di mosse considerate K(t). Bilancia il costo computazionale: cerca in profondità solo sulle mosse più promettenti.\n\nL(t): Crescente\nK(t): Decrescente",
            'exponential': "STRATEGIA: Crescita Esponenziale [L(t), K(t)]\n\nLa profondità di ricerca L(t) aumenta esponenzialmente. L'agente sfrutta il ridotto fattore di branching nelle fasi finali.\n\nL(t): Crescita Exp\nK(t): Decadimento Lineare",
            'sigmoid': "STRATEGIA: Crescita Sigmoide [L(t), K(t)]\n\nL(t) e K(t) variano seguendo una curva a S. Transizione rapida da 'esploratore' a 'calcolatore profondo' a metà partita.\n\nL(t), K(t): Curva Sigmoide",
            'constant': "BASELINE: Parametri Costanti [L, K]\n\nL'agente usa parametri fissi per tutta la partita. Utile come riferimento per valutare l'efficacia delle strategie adattive.\n\nL, K: Costanti",
            'staircase': "STRATEGIA: A Gradini (Staircase) [L(t), K(t)]\n\nI parametri cambiano a scatti (gradini). Permette all'agente di stabilizzarsi su un livello di complessità prima di passare al successivo.\n\nL(t), K(t): A Gradini"
        }
        
        # Description Box
        self.desc_rect = pygame.Rect(400, 150, 400, 300)
        
        # Theory Info Box (Bottom Right)
        self.theory_rect = pygame.Rect(400, 470, 400, 100)
        self.theory_text = [
            "Predictive MinMax (HW1 AI 25/26):",
            "L'agente usa action(s) := MinMax(s, Htrue, L, K)",
            "Htrue è un MLP addestrato via Self-Play.",
            "L=Depth Cut, K=Width Cut."
        ]

    def start_game(self):
        grid_size = int(self.grid_size_sel.options[self.grid_size_sel.selected_index][0])
        p1_type = self.p1_type_sel.options[self.p1_type_sel.selected_index]
        p2_type = self.p2_type_sel.options[self.p2_type_sel.selected_index]
        strategy = self.strategy_sel.options[self.strategy_sel.selected_index]
        
        print(f"Starting game: Grid={grid_size}, P1={p1_type}, P2={p2_type}, Strategy={strategy}")
        self.app.start_game(grid_size, p1_type, p2_type, strategy)

    def open_benchmark(self):
        self.app.open_benchmark()

    def draw(self, screen):
        screen.fill(LIGHT_GRAY)
        
        title_font = Theme.get_font(48, bold=True)
        title_surf = title_font.render("Predictive MinMax - Homework 1", True, Theme.PRIMARY)
        subtitle_font = Theme.get_font(24)
        subtitle_surf = subtitle_font.render("Artificial Intelligence 25/26", True, Theme.SECONDARY)
        
        title_rect = title_surf.get_rect(center=(WINDOW_WIDTH // 2, 50))
        subtitle_rect = subtitle_surf.get_rect(center=(WINDOW_WIDTH // 2, 90))
        
        screen.blit(title_surf, title_rect)
        screen.blit(subtitle_surf, subtitle_rect)

        self.grid_size_sel.draw(screen)
        self.p1_type_sel.draw(screen)
        self.p2_type_sel.draw(screen)
        
        # Only show strategy if at least one player is AI
        show_strategy = "AI" in [self.p1_type_sel.options[self.p1_type_sel.selected_index], 
                                self.p2_type_sel.options[self.p2_type_sel.selected_index]]
        
        if show_strategy:
            self.strategy_sel.draw(screen)
            
            # Draw description box
            # Shadow
            shadow_rect = self.desc_rect.copy()
            shadow_rect.y += 4
            pygame.draw.rect(screen, Theme.SHADOW, shadow_rect, border_radius=8)
            
            pygame.draw.rect(screen, Theme.WHITE, self.desc_rect, border_radius=8)
            pygame.draw.rect(screen, Theme.GRAY, self.desc_rect, 1, border_radius=8)
            
            # Draw description title
            font = Theme.get_font(32, bold=True)
            title = font.render("Dettagli Strategia", True, Theme.PRIMARY)
            screen.blit(title, (self.desc_rect.x + 20, self.desc_rect.y + 15))
            pygame.draw.line(screen, Theme.GRAY, (self.desc_rect.x, self.desc_rect.y + 50), (self.desc_rect.right, self.desc_rect.y + 50), 1)
            
            # Draw description text
            selected_strategy = self.strategy_sel.options[self.strategy_sel.selected_index]
            desc_text = self.strategy_descriptions.get(selected_strategy, "")
            
            # Word wrap
            font_small = pygame.font.Font(None, 28)
            y_offset = 60
            
            # Split by newlines first to respect manual formatting
            paragraphs = desc_text.split('\n')
            for paragraph in paragraphs:
                if not paragraph:
                    y_offset += 15 # Small gap for empty lines
                    continue
                    
                # Wrap each paragraph
                wrapped_lines = self.wrap_text(paragraph, font_small, 380)
                for line in wrapped_lines:
                    surf = font_small.render(line, True, BLACK)
                    screen.blit(surf, (self.desc_rect.x + 10, self.desc_rect.y + y_offset))
                    y_offset += 25

                    y_offset += 25
 
            # Draw Theory Info
            # Shadow
            shadow_rect = self.theory_rect.copy()
            shadow_rect.y += 4
            pygame.draw.rect(screen, Theme.SHADOW, shadow_rect, border_radius=8)
            
            pygame.draw.rect(screen, Theme.WHITE, self.theory_rect, border_radius=8)
            pygame.draw.rect(screen, Theme.GRAY, self.theory_rect, 1, border_radius=8)
            
            font_theory = Theme.get_font(20)
            for i, line in enumerate(self.theory_text):
                color = Theme.PRIMARY if i == 0 else Theme.TEXT
                surf = font_theory.render(line, True, color)
                screen.blit(surf, (self.theory_rect.x + 15, self.theory_rect.y + 10 + i * 22))

        self.start_btn.draw(screen)

    def wrap_text(self, text, font, max_width):
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            w, h = font.size(test_line)
            if w < max_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        
        lines.append(' '.join(current_line))
        return lines

    def handle_event(self, event):
        if self.grid_size_sel.handle_event(event): return
        if self.p1_type_sel.handle_event(event): return
        if self.p2_type_sel.handle_event(event): return
        
        if "AI" in [self.p1_type_sel.options[self.p1_type_sel.selected_index], 
                   self.p2_type_sel.options[self.p2_type_sel.selected_index]]:
            if self.strategy_sel.handle_event(event): return

        self.start_btn.handle_event(event)


class BenchmarkView:
    def __init__(self, app):
        self.app = app
        self.back_btn = Button(10, WINDOW_HEIGHT - 60, 150, 40, "Indietro", self.app.to_menu)
        self.font_title = pygame.font.Font(None, 48)
        self.font_text = pygame.font.Font(None, 28)
        
        # Static Benchmark Data (Theoretical/Experimental)
        self.data = [
            ("Strategia", "Win Rate (vs Rand)", "Tempo Medio", "Complessità"),
            ("Progressive", "85%", "Basso", "Crescente"),
            ("Inverse", "82%", "Medio", "Bilanciata"),
            ("Exponential", "88%", "Alto (Finale)", "Esponenziale"),
            ("Sigmoid", "86%", "Medio", "Variabile"),
            ("Staircase", "84%", "Basso", "A Gradini"),
            ("Constant", "75%", "Costante", "Fissa")
        ]

    def draw(self, screen):
        screen.fill(Theme.BG)
        
        # Title
        title = self.font_title.render("Benchmark Modelli AI", True, Theme.PRIMARY)
        screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 50))
        
        # Disclaimer
        disc_font = Theme.get_font(20, bold=True)
        disc = disc_font.render("* Dati stimati. Esegui experiment.ipynb per i risultati reali.", True, Theme.RED)
        screen.blit(disc, (WINDOW_WIDTH // 2 - disc.get_width() // 2, 100))
        
        # Table
        start_y = 150
        col_widths = [200, 200, 200, 200]
        start_x = (WINDOW_WIDTH - sum(col_widths)) // 2
        
        for i, row in enumerate(self.data):
            y = start_y + i * 50
            x = start_x
            
            # Draw row background
            if i == 0:
                pygame.draw.rect(screen, DARK_GRAY, (x, y, sum(col_widths), 40))
            elif i % 2 == 0:
                pygame.draw.rect(screen, WHITE, (x, y, sum(col_widths), 40))
            else:
                pygame.draw.rect(screen, (240, 240, 240), (x, y, sum(col_widths), 40))
                
            for j, text in enumerate(row):
                color = Theme.WHITE if i == 0 else Theme.TEXT
                surf = self.font_text.render(text, True, color)
                rect = surf.get_rect(center=(x + col_widths[j] // 2, y + 20))
                screen.blit(surf, rect)
                x += col_widths[j]
                
        self.back_btn.draw(screen)

    def handle_event(self, event):
        self.back_btn.handle_event(event)

class Game:
    def __init__(self, app, grid_size, p1_type, p2_type, strategy_name):
        self.app = app
        self.grid_size = grid_size
        self.p1_type = p1_type
        self.p2_type = p2_type
        self.strategy_name = strategy_name
        
        self.game = DotsAndBoxes(grid_size=grid_size)
        
        # Initialize AI
        self.mlp = MLPEvaluator.create_from_game(self.game, hidden_sizes=[64, 32])
        strategies = get_all_strategies()
        strategy = strategies[strategy_name]
        L, K = strategy.get_params(0) # Use initial params
        self.ai_agent = MinMaxAgent(self.mlp, L=L, K=K)
        
        self.game_over = False
        self.ai_thinking = False
        self.winner = None
        
        # Layout
        GAME_AREA_WIDTH = 800
        grid_width = (grid_size) * DOT_SPACING
        grid_height = (grid_size) * DOT_SPACING
        self.offset_x = (GAME_AREA_WIDTH - grid_width) // 2
        self.offset_y = (WINDOW_HEIGHT - grid_height) // 2
        
        self.back_btn = Button(10, WINDOW_HEIGHT - 60, 150, 40, "Back to Menu", self.app.to_menu)
        self.font = pygame.font.Font(None, 36)
        
        # Initialize Log (Resized for Rules)
        self.log = ScrollableGameLog(750, 20, 230, 350) # Adjusted position and size
        self.log.add_message(f"Partita Iniziata: {grid_size}x{grid_size}")
        self.log.add_message(f"P1: {p1_type}, P2: {p2_type}")
        self.log.add_message(f"Strategia: {strategy_name}")
        self.log.add_message(f"Parametri Iniziali:")
        self.log.add_message(f"  L={L}, K={K}")
        
        # AI Parameter Visualization
        self.current_L = L
        self.current_K = K
        self.move_count = 0

    def draw(self, screen):
        screen.fill(Theme.BG)
        self.draw_boxes(screen)
        self.draw_lines(screen)
        self.draw_dots(screen)
        self.draw_ui(screen)
        self.draw_hover(screen)
        self.back_btn.draw(screen)
        self.log.draw(screen)
        self.draw_rules(screen)

    def draw_rules(self, screen):
        # Rules Box below Log
        rect = pygame.Rect(750, 390, 230, 180) # Adjusted position and size
        
        # Shadow
        shadow_rect = rect.copy()
        shadow_rect.y += 2
        pygame.draw.rect(screen, Theme.SHADOW, shadow_rect, border_radius=5)
        
        pygame.draw.rect(screen, Theme.WHITE, rect, border_radius=5)
        pygame.draw.rect(screen, Theme.GRAY, rect, 1, border_radius=5)
        
        # Title
        font_title = Theme.get_font(24, bold=True)
        title = font_title.render("REGOLAMENTO", True, Theme.PRIMARY)
        screen.blit(title, (rect.x + 10, rect.y + 10))
        pygame.draw.line(screen, Theme.GRAY, (rect.x, rect.y + 35), (rect.right, rect.y + 35), 1)
        
        # Rules text
        rules = [
            "1. Clicca tra due punti",
            "   per creare una linea.",
            "2. Chi chiude un box",
            "   fa punto e rigioca.",
            "3. Vince chi ha più",
            "   box alla fine.",
            "4. Rosso=P1, Blu=P2"
        ]
        font_rules = Theme.get_font(20)
        for i, rule in enumerate(rules):
            surf = font_rules.render(rule, True, Theme.TEXT)
            screen.blit(surf, (rect.x + 10, rect.y + 45 + i * 20))

    def draw_dots(self, screen):
        for r in range(self.grid_size + 1):
            for c in range(self.grid_size + 1):
                x = self.offset_x + c * DOT_SPACING
                y = self.offset_y + r * DOT_SPACING
                pygame.draw.circle(screen, BLACK, (x, y), DOT_RADIUS)

    def draw_lines(self, screen):
        # Horizontal edges
        for r in range(self.grid_size + 1):
            for c in range(self.grid_size):
                owner = self.game.horizontal_edges[r, c]
                if owner != 0:
                    color = RED if owner == 1 else BLUE
                    start_pos = (self.offset_x + c * DOT_SPACING, self.offset_y + r * DOT_SPACING)
                    end_pos = (self.offset_x + (c + 1) * DOT_SPACING, self.offset_y + r * DOT_SPACING)
                    pygame.draw.line(screen, color, start_pos, end_pos, LINE_WIDTH)

        # Vertical edges
        for r in range(self.grid_size):
            for c in range(self.grid_size + 1):
                owner = self.game.vertical_edges[r, c]
                if owner != 0:
                    color = RED if owner == 1 else BLUE
                    start_pos = (self.offset_x + c * DOT_SPACING, self.offset_y + r * DOT_SPACING)
                    end_pos = (self.offset_x + c * DOT_SPACING, self.offset_y + (r + 1) * DOT_SPACING)
                    pygame.draw.line(screen, color, start_pos, end_pos, LINE_WIDTH)

    def draw_boxes(self, screen):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                owner = self.game.boxes[r, c]
                if owner != 0:
                    x = self.offset_x + c * DOT_SPACING + LINE_WIDTH // 2
                    y = self.offset_y + r * DOT_SPACING + LINE_WIDTH // 2
                    width = DOT_SPACING - LINE_WIDTH
                    height = DOT_SPACING - LINE_WIDTH
                    color = RED if owner == 1 else BLUE
                    pygame.draw.rect(screen, color, (x, y, width, height))

        # Draw Rules (Moved to separate method)
        pass

    def draw_hover(self, screen):
        if self.game_over or self.ai_thinking: return
        
        # Only show hover for human players
        current_is_human = (self.game.current_player == 1 and self.p1_type == "Human") or \
                           (self.game.current_player == 2 and self.p2_type == "Human")
        
        if not current_is_human: return

        mouse_pos = pygame.mouse.get_pos()
        move = self.get_clicked_line(mouse_pos)
        
        if move and move in self.game.get_valid_moves():
            orientation, r, c = move
            if orientation == 'h': # Horizontal
                start_pos = (self.offset_x + c * DOT_SPACING, self.offset_y + r * DOT_SPACING)
                end_pos = (self.offset_x + (c + 1) * DOT_SPACING, self.offset_y + r * DOT_SPACING)
            else: # Vertical
                start_pos = (self.offset_x + c * DOT_SPACING, self.offset_y + r * DOT_SPACING)
                end_pos = (self.offset_x + c * DOT_SPACING, self.offset_y + (r + 1) * DOT_SPACING)
            
            # Draw ghost line
            pygame.draw.line(screen, (200, 200, 0), start_pos, end_pos, LINE_WIDTH)

        if DEBUG_HITBOXES:
            # Draw all hitboxes
            # Horizontal
            for r in range(self.grid_size + 1):
                for c in range(self.grid_size):
                    x1 = self.offset_x + c * DOT_SPACING
                    y1 = self.offset_y + r * DOT_SPACING - HITBOX_WIDTH // 2
                    rect = pygame.Rect(x1, y1, DOT_SPACING, HITBOX_WIDTH)
                    pygame.draw.rect(screen, (0, 255, 0), rect, 1)
            
            # Vertical
            for r in range(self.grid_size):
                for c in range(self.grid_size + 1):
                    x1 = self.offset_x + c * DOT_SPACING - HITBOX_WIDTH // 2
                    y1 = self.offset_y + r * DOT_SPACING
                    rect = pygame.Rect(x1, y1, HITBOX_WIDTH, DOT_SPACING)
                    pygame.draw.rect(screen, (0, 0, 255), rect, 1)

    def draw_ui(self, screen):
        # Player turn
        p1_text = f"P1 (Red): {self.p1_type}"
        p2_text = f"P2 (Blue): {self.p2_type}"
        
        turn_color = Theme.RED if self.game.current_player == 1 else Theme.BLUE
        turn_text = f"Turn: {'Player 1' if self.game.current_player == 1 else 'Player 2'}"
        
        screen.blit(self.font.render(p1_text, True, Theme.RED), (10, 10))
        screen.blit(self.font.render(p2_text, True, Theme.BLUE), (10, 40))
        screen.blit(self.font.render(turn_text, True, turn_color), (10, 80))
        
        # Scores
        score_text = f"Red: {self.game.player1_score}  Blue: {self.game.player2_score}"
        screen.blit(self.font.render(score_text, True, Theme.TEXT), (WINDOW_WIDTH - 300, 10))
        
        if self.ai_thinking:
            screen.blit(self.font.render("AI thinking...", True, GRAY), (WINDOW_WIDTH - 200, 50))

        if self.game_over:
            winner_text = "Draw!"
            color = GREEN
            if self.game.player1_score > self.game.player2_score:
                winner_text = "Player 1 Wins!"
                color = RED
            elif self.game.player2_score > self.game.player1_score:
                winner_text = "Player 2 Wins!"
                color = BLUE
            
            text = self.font.render(f"GAME OVER: {winner_text}", True, color)
            rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 100))
            pygame.draw.rect(screen, WHITE, rect.inflate(20, 20))
            pygame.draw.rect(screen, BLACK, rect.inflate(20, 20), 2)
            screen.blit(text, rect)

    def get_clicked_line(self, pos):
        x, y = pos
        # Check horizontal lines
        for r in range(self.grid_size + 1):
            for c in range(self.grid_size):
                x1 = self.offset_x + c * DOT_SPACING
                y1 = self.offset_y + r * DOT_SPACING - HITBOX_WIDTH // 2
                rect = pygame.Rect(x1, y1, DOT_SPACING, HITBOX_WIDTH)
                if rect.collidepoint(x, y):
                    return ('h', r, c)
        
        # Check vertical lines
        for r in range(self.grid_size):
            for c in range(self.grid_size + 1):
                x1 = self.offset_x + c * DOT_SPACING - HITBOX_WIDTH // 2
                y1 = self.offset_y + r * DOT_SPACING
                rect = pygame.Rect(x1, y1, HITBOX_WIDTH, DOT_SPACING)
                if rect.collidepoint(x, y):
                    return ('v', r, c)
        return None

    def ai_move_thread(self):
        time.sleep(0.5) # Small delay for visual clarity
        
        # Update AI parameters based on strategy and move count
        # We map move_count to 'iteration' to demonstrate the strategy
        strategies = get_all_strategies()
        strategy = strategies[self.strategy_name]
        
        # Scale move count to simulate training iterations for demonstration
        # E.g., 1 game move = 2 'iterations' of strategy evolution
        simulated_iteration = self.move_count * 2
        L, K = strategy.get_params(simulated_iteration)
        
        # Log parameter changes with explanation
        if L != self.current_L or K != self.current_K or self.move_count == 0:
            explanation = ""
            if self.strategy_name == 'progressive':
                explanation = "(Aumento profondità)"
            elif self.strategy_name == 'inverse':
                explanation = "(Più profondo, meno mosse)"
            elif self.strategy_name == 'exponential':
                explanation = "(Crescita esponenziale)"
            elif self.strategy_name == 'sigmoid':
                explanation = "(Transizione sigmoide)"
            elif self.strategy_name == 'staircase':
                explanation = "(Livello completato)"
            
            self.log.add_message(f"Turno {self.move_count}: L={L}, K={K}")
            if explanation:
                self.log.add_message(f"  {explanation}")

        self.ai_agent.update_parameters(L, K)
        self.current_L = L
        self.current_K = K
        
        move = self.ai_agent.select_move(self.game)
        # self.log.add_message(f"AI (L={L}, K={K}) gioca: {move}") # Removed move log
        self.game.make_move(move)
        self.move_count += 1
        self.ai_thinking = False
        self.check_game_over()

    def check_game_over(self):
        if self.game.is_game_over():
            self.game_over = True
            winner = self.game.get_winner()
            if winner == 1:
                self.log.add_message("GAME OVER: Player 1 Wins!")
            elif winner == -1:
                self.log.add_message("GAME OVER: Player 2 Wins!")
            else:
                self.log.add_message("GAME OVER: Draw!")

    def update(self):
        if self.game_over: return

        current_player_type = self.p1_type if self.game.current_player == 1 else self.p2_type
        
        if current_player_type == "AI" and not self.ai_thinking:
            self.ai_thinking = True
            threading.Thread(target=self.ai_move_thread, daemon=True).start()

    def handle_event(self, event):
        self.back_btn.handle_event(event)
        
        if self.game_over: return

        current_player_type = self.p1_type if self.game.current_player == 1 else self.p2_type
        
        if current_player_type == "Human" and not self.ai_thinking:
            if event.type == pygame.MOUSEBUTTONDOWN:
                move = self.get_clicked_line(event.pos)
                if move:
                    if move in self.game.get_valid_moves():
                        # self.log.add_message(f"P{self.game.current_player} gioca: {move}") # Removed move log
                        self.game.make_move(move)
                        self.move_count += 1
                        self.check_game_over()
                    else:
                        self.log.add_message("Mossa non valida!")
        
        # Handle log scrolling
        self.log.handle_scroll(event)

class DotsAndBoxesGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Dots and Boxes - Predictive MinMax")
        self.clock = pygame.time.Clock()
        
        self.state = "MENU" # MENU, GAME, BENCHMARK
        self.menu = MainMenu(self)
        self.game = None
        self.benchmark = BenchmarkView(self)

    def start_game(self, grid_size, p1_type, p2_type, strategy):
        self.game = Game(self, grid_size, p1_type, p2_type, strategy)
        self.state = "GAME"

    def open_benchmark(self):
        self.state = "BENCHMARK"

    def to_menu(self):
        self.state = "MENU"
        self.game = None

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if self.state == "MENU":
                    self.menu.handle_event(event)
                elif self.state == "GAME":
                    self.game.handle_event(event)
                elif self.state == "BENCHMARK":
                    self.benchmark.handle_event(event)

            if self.state == "MENU":
                self.menu.draw(self.screen)
            elif self.state == "GAME":
                self.game.update()
                self.game.draw(self.screen)
            elif self.state == "BENCHMARK":
                self.benchmark.draw(self.screen)
            
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    gui = DotsAndBoxesGUI()
    gui.run()
