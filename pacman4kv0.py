# --------------------------------------------
# Enhanced Pac-Man Arcade Game - test.py
# Faithful recreation with programmatic audio
# --------------------------------------------
import pygame
import random
import math
import numpy as np

# Initialize Pygame and audio
pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Game Constants
TILE_SIZE = 20
MAZE_COLS = 28
MAZE_ROWS = 31
SCREEN_WIDTH = TILE_SIZE * MAZE_COLS
SCREEN_HEIGHT = TILE_SIZE * MAZE_ROWS + 100 # Extra space for UI
FPS = 60

# Colors (matching original arcade)
BLACK = (0, 0, 0)
# BLUE = (33, 33, 255) # Original wall color in some versions
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PINK = (255, 184, 255)
CYAN = (0, 255, 255)
ORANGE = (255, 184, 82)
WHITE = (255, 255, 255)
BLUE_GHOST_FRIGHTENED = (33, 33, 222) # For frightened ghosts
WALL_BLUE = (33, 33, 255) # Arcade accurate wall color

# Game Settings
START_LIVES = 3
DOT_SCORE = 10
POWER_DOT_SCORE = 50
GHOST_EAT_SCORES = [200, 400, 800, 1600]
# FRUIT_VALUES = [100, 300, 500, 700, 1000, 2000, 3000, 5000] # For later
POWER_PELLET_TIME = 6 * FPS  # 6 seconds of fright
READY_TIME = int(2.0 * FPS) # 2 seconds ready screen, adjusted for clarity
PACMAN_DEATH_TIME = int(2.0 * FPS) # Duration of Pac-Man's death animation

# Maze layout (Standard Pac-Man Maze)
maze_layout = [
    "############################",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#o####.#####.##.#####.####o#", # o for power pellet
    "#.####.#####.##.#####.####.#",
    "#..........................#",
    "#.####.##.########.##.####.#",
    "#.####.##.########.##.####.#",
    "#......##....##....##......#",
    "######.##### ## #####.######", # Space for tunnel exit alignment
    "######.##### ## #####.######",
    "######.##    G   ##.######", # G is ghost house area, but accessible
    "######.## ###--### ##.######", # -- is ghost house door
    "######.## #      # ##.######", # Ghost Pen
    "       .   #      #   .      ", # Tunnel with dots
    "######.## #      # ##.######",
    "######.## ######## ##.######",
    "######.##    P   ##.######", # P for Pacman Start reference, not used directly by maze draw
    "######.## ######## ##.######",
    "######.## ######## ##.######",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#o..##................##..o#",
    "###.##.##.########.##.##.###",
    "###.##.##.########.##.##.###",
    "#......##....##....##......#",
    "#.##########.##.##########.#",
    "#.##########.##.##########.#",
    "#..........................#",
    "############################"
]

# Ghost spawn and target locations
GHOST_START_POSITIONS = {
    "blinky": (13.5, 11), # Starts outside, above pen
    "pinky": (13.5, 14),  # Inside pen, center
    "inky": (11.5, 14),    # Inside pen, left
    "clyde": (15.5, 14)   # Inside pen, right
}
GHOST_COLORS = {"blinky": RED, "pinky": PINK, "inky": CYAN, "clyde": ORANGE}

# Scatter targets (tile coordinates, often outside maze for pathing)
SCATTER_TARGETS = {
    "blinky": (MAZE_COLS - 2, -2),      # Top-right
    "pinky": (1, -2),                  # Top-left
    "inky": (MAZE_COLS - 1, MAZE_ROWS),# Bottom-right
    "clyde": (0, MAZE_ROWS)            # Bottom-left
}
GHOST_HOME_EXIT_TILE = (13, 11) # Tile position to exit pen (using column 13, row 11 from example, should be 13.5 for true center?)
                                # For tile coordinates, integer values are usually preferred.
GHOST_PEN_CENTER_TILE = (13.5, 14) # Used for returning dead ghosts

# Audio Generation Functions (minor adjustments for clarity/consistency)
def generate_tone(frequency, duration, sample_rate=22050, amplitude=0.1):
    frames = int(duration * sample_rate)
    t = np.linspace(0, duration, frames, endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    # Ensure stereo
    stereo_wave = np.zeros((frames, 2))
    stereo_wave[:, 0] = wave
    stereo_wave[:, 1] = wave
    sound_array = (stereo_wave * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(sound_array)

def generate_waka_sound(): # More of a short blip, good for dot eating
    return generate_tone(random.randint(300, 500), 0.05, amplitude=0.08)


def generate_power_pellet_sound(): # Siren-like loop for power mode
    sample_rate = 22050
    duration = 0.5 # Loop duration
    frames = int(duration * sample_rate)
    arr = np.zeros((frames, 2))
    for i in range(frames):
        freq = 600 + 200 * math.sin(2 * math.pi * 4 * i / frames) # Modulated frequency
        arr[i, 0] = 0.08 * math.sin(2 * math.pi * freq * i / sample_rate)
    arr[:, 1] = arr[:, 0]
    arr = (arr * 32767).astype(np.int16)
    sound = pygame.sndarray.make_sound(arr)
    sound.set_volume(0.5)
    return sound

def generate_ghost_eaten_sound():
    return generate_tone(1200, 0.2, amplitude=0.12)

def generate_death_sound():
    sample_rate = 22050
    duration = 1.5 # Longer, more dramatic
    frames = int(duration * sample_rate)
    arr = np.zeros((frames, 2))
    start_freq, end_freq = 800, 50
    for i in range(frames):
        progress = i / frames
        freq = start_freq * (1 - progress) + end_freq * progress # Descending sweep
        amplitude = 0.15 * (1 - progress**2)  # Fade out more sharply
        # Add some vibrato/warble for effect
        warble = 1 + 0.1 * math.sin(2 * math.pi * 10 * progress)
        arr[i, 0] = amplitude * math.sin(2 * np.pi * freq * warble * i / sample_rate)
    arr[:, 1] = arr[:, 0]
    arr = (arr * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(arr)

def generate_game_start_sound():
    return generate_tone(440, 0.5, amplitude=0.1)


# Initialize sounds
sounds = {
    'chomp': generate_waka_sound(), # Using waka as chomp
    'power_pellet_active': generate_power_pellet_sound(), # Looping siren for power mode
    'ghost_eaten': generate_ghost_eaten_sound(),
    'death': generate_death_sound(),
    'game_start': generate_game_start_sound(),
    'extra_life': generate_tone(880, 0.3, amplitude=0.1) # Placeholder
}

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("PAC-MAN by Cat-san")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36) # Default font
        self.small_font = pygame.font.Font(None, 28)

        self.high_score = 0 # TODO: Load/save high score
        self.init_game_session()

    def init_game_session(self):
        self.state = "START_SCREEN" # START_SCREEN, READY, PLAYING, DYING, GAME_OVER, LEVEL_CLEAR, GHOST_EAT_PAUSE
        self.state_timer = 0
        self.score = 0
        self.lives = START_LIVES
        self.level = 1
        self.pellets_eaten_total_level = 0 # Dots eaten in current level configuration
        self.total_pellets_in_maze = self.count_pellets_in_layout()
        self.ghost_eat_combo_count = 0 # For sequential ghost eating scores
        self.power_mode_active = False
        self.power_mode_timer = 0
        self.extra_life_awarded_at_10k = False

        self.maze = [list(row) for row in maze_layout] # Current mutable maze
        self.original_maze = [list(row) for row in maze_layout] # For reset

        self.pacman = Pacman(self, (13.5, 23)) # Pacman start tile (col, row)
        self.ghosts = [] # Will be populated by setup_level_entities

        # Ghost mode cycling (scatter/chase) - durations in seconds
        self.ghost_wave_patterns = {
            1: [(7, 20), (7, 20), (5, 20), (5, float('inf'))], # Level 1: Scatter (7s), Chase (20s), S(7), C(20), S(5), C(20), S(5), C(inf)
            # Add more for other levels
        }
        self.current_ghost_wave = 0
        self.ghost_mode = "scatter" # Initial mode
        self.ghost_mode_timer = 0 # Set in reset_level or start_new_level
        self.setup_level_entities() # Call after ghost_mode_patterns is set

        self.power_pellet_blink_timer = 0

    def start_new_game(self):
        self.high_score = max(self.high_score, self.score)
        self.init_game_session() # This will re-initialize everything including entities
        self.state = "READY"
        self.state_timer = READY_TIME
        sounds['game_start'].play()
        self.reset_level_soft() # Soft reset for positions and ghost modes

    def count_pellets_in_layout(self):
        count = 0
        for r_idx, row_str in enumerate(maze_layout):
            for c_idx, char_in_row in enumerate(row_str):
                if char_in_row == '.' or char_in_row == 'o':
                    count += 1
        return count

    def setup_level_entities(self):
        self.pacman.reset_position()
        # Clear pellet under Pac-Man's start
        px, py = self.pacman.get_tile()
        if 0 <= py < MAZE_ROWS and 0 <= px < MAZE_COLS:
            if self.maze[py][px] in ['.', 'o']:
                self.maze[py][px] = ' '

        self.ghosts = [
            Ghost(self, "blinky", GHOST_START_POSITIONS["blinky"], GHOST_COLORS["blinky"], SCATTER_TARGETS["blinky"]),
            Ghost(self, "pinky", GHOST_START_POSITIONS["pinky"], GHOST_COLORS["pinky"], SCATTER_TARGETS["pinky"]),
            Ghost(self, "inky", GHOST_START_POSITIONS["inky"], GHOST_COLORS["inky"], SCATTER_TARGETS["inky"]),
            Ghost(self, "clyde", GHOST_START_POSITIONS["clyde"], GHOST_COLORS["clyde"], SCATTER_TARGETS["clyde"])
        ]
        self.blinky_ref = self.ghosts[0] # For Inky's targeting

        # Ghost release counters/logic
        self.dots_for_ghost_release = { "pinky": 0, "inky": 30, "clyde": 60 }
        self.pellets_eaten_this_life = 0
        self.reset_ghost_modes_and_timers() # Initialize timers correctly


    def reset_level_hard(self): # After game over, or advancing level
        self.maze = [list(row) for row in self.original_maze]
        self.pellets_eaten_total_level = 0
        # self.total_pellets_in_maze = self.count_pellets_in_layout() # Recount if maze can change, usually static
        self.setup_level_entities() # This re-creates ghosts and resets Pacman
        # self.reset_ghost_modes_and_timers() # Already called in setup_level_entities
        self.pellets_eaten_this_life = 0
        self.power_mode_active = False
        self.power_mode_timer = 0
        if sounds['power_pellet_active'].get_num_channels() > 0:
            sounds['power_pellet_active'].stop()


    def reset_level_soft(self): # After Pac-Man loses a life
        self.pacman.reset_position()
        for ghost in self.ghosts:
            ghost.reset_to_start_or_pen(self.ghost_mode) # Pass current game mode for Blinky
        self.reset_ghost_modes_and_timers() # Reset scatter/chase sequence
        self.power_mode_active = False
        self.power_mode_timer = 0
        if sounds['power_pellet_active'].get_num_channels() > 0:
            sounds['power_pellet_active'].stop()
        self.pellets_eaten_this_life = 0 # Reset for ghost release


    def reset_ghost_modes_and_timers(self):
        self.current_ghost_wave = 0
        wave_pattern = self.ghost_wave_patterns.get(self.level, self.ghost_wave_patterns[1])
        self.ghost_mode = "scatter" # Start with scatter
        self.ghost_mode_timer = wave_pattern[self.current_ghost_wave][0] * FPS
        for ghost in self.ghosts: # Ensure ghosts adopt current mode if not in pen/dead/frightened
             if ghost.mode not in ["pen", "leaving_pen", "dead", "entering_pen", "frightened"]:
                ghost.set_mode(self.ghost_mode) # This will also trigger reversal flag


    def handle_input(self):
        keys = pygame.key.get_pressed()
        if self.state == "PLAYING":
            new_dir = None
            if keys[pygame.K_UP]: new_dir = (0, -1)
            elif keys[pygame.K_DOWN]: new_dir = (0, 1)
            elif keys[pygame.K_LEFT]: new_dir = (-1, 0)
            elif keys[pygame.K_RIGHT]: new_dir = (1, 0)
            if new_dir:
                self.pacman.set_next_direction(new_dir)

    def update(self):
        if self.state_timer > 0: # Only decrement if timer is active
            self.state_timer -= 1

        self.power_pellet_blink_timer = (self.power_pellet_blink_timer + 1) % 20 # For blinking power pellets

        if self.state == "START_SCREEN":
            pass # Handled by event loop for key press
        elif self.state == "READY":
            if self.state_timer <= 0:
                self.state = "PLAYING"
        elif self.state == "PLAYING":
            self.update_game_play()
        elif self.state == "DYING":
            if self.state_timer <= 0:
                self.lives -= 1
                if self.lives <= 0:
                    self.state = "GAME_OVER"
                    self.state_timer = 180 # Display for 3 seconds
                    if sounds['power_pellet_active'].get_num_channels() > 0:
                        sounds['power_pellet_active'].stop()
                else:
                    self.state = "READY"
                    self.state_timer = READY_TIME
                    self.reset_level_soft()
        elif self.state == "GAME_OVER":
            if self.state_timer <= 0:
                self.state = "START_SCREEN"
        elif self.state == "LEVEL_CLEAR":
            if self.state_timer <= 0:
                self.level += 1
                self.state = "READY"
                self.state_timer = READY_TIME
                self.reset_level_hard()
                self.pacman.speed = min(2.5, self.pacman.speed + 0.05) # Slightly increase Pac-Man speed per level
                for ghost in self.ghosts: ghost.speed = min(2.0, ghost.speed + 0.025)
        elif self.state == "GHOST_EAT_PAUSE":
            if self.state_timer <= 0:
                self.state = "PLAYING"


    def update_ghost_mode_cycle(self):
        if self.power_mode_active: # Scatter/chase timer paused during fright
            return

        if self.ghost_mode_timer == float('inf'): # Infinite mode, timer doesn't count down
            return

        self.ghost_mode_timer -= 1
        if self.ghost_mode_timer <= 0:
            wave_pattern = self.ghost_wave_patterns.get(self.level, self.ghost_wave_patterns[1])
            
            if self.ghost_mode == "scatter":
                self.ghost_mode = "chase"
                self.ghost_mode_timer = wave_pattern[self.current_ghost_wave][1] * FPS
            else: # Was chase, so chase timer ended. Switch to SCATTER of next wave.
                self.current_ghost_wave += 1
                if self.current_ghost_wave >= len(wave_pattern):
                    # All defined waves are done. Pin to the last wave's pattern.
                    # If last chase was finite, this will loop the last wave's scatter/chase.
                    self.current_ghost_wave = len(wave_pattern) - 1 
                    self.ghost_mode = "scatter" # Re-start scatter of this pinned wave
                    self.ghost_mode_timer = wave_pattern[self.current_ghost_wave][0] * FPS
                else:
                    # Normal transition to next scatter wave
                    self.ghost_mode = "scatter"
                    self.ghost_mode_timer = wave_pattern[self.current_ghost_wave][0] * FPS
            
            # Tell ghosts to update their mode if they are active and not frightened
            for ghost in self.ghosts:
                if ghost.mode not in ["pen", "leaving_pen", "dead", "entering_pen", "frightened"]:
                    ghost.set_mode(self.ghost_mode) # This also sets reverse_direction_at_next_intersection


    def update_game_play(self):
        self.pacman.update(self.maze)
        self.handle_pellet_eating()
        self.update_power_mode()
        self.update_ghost_mode_cycle()
        self.release_ghosts_from_pen()

        for ghost in self.ghosts:
            ghost.update(self.pacman, self.blinky_ref, self.maze, self.ghost_mode)

        self.check_collisions() # This can change game state to DYING or GHOST_EAT_PAUSE
        if self.state == "PLAYING": # Only check level completion if still playing
            self.check_level_completion()

        if not self.extra_life_awarded_at_10k and self.score >= 10000:
            self.lives += 1
            self.extra_life_awarded_at_10k = True
            sounds['extra_life'].play()


    def handle_pellet_eating(self):
        pac_tile_x, pac_tile_y = self.pacman.get_tile()
        # Ensure Pacman is centered enough over a tile to eat pellet (prevents eating from edge of tile)
        center_x_tile, center_y_tile = self.pacman.get_pixel_pos_snapped_to_tile_center(pac_tile_x, pac_tile_y)
        if abs(self.pacman.x - center_x_tile) > TILE_SIZE * 0.4 or abs(self.pacman.y - center_y_tile) > TILE_SIZE * 0.4:
            return # Not centered enough

        if 0 <= pac_tile_y < MAZE_ROWS and 0 <= pac_tile_x < MAZE_COLS:
            cell = self.maze[pac_tile_y][pac_tile_x]
            if cell == '.':
                self.maze[pac_tile_y][pac_tile_x] = ' '
                self.score += DOT_SCORE
                self.pellets_eaten_total_level += 1
                self.pellets_eaten_this_life += 1
                sounds['chomp'].play()
                self.pacman.is_eating_dot_timer = 2 # For speed adjustment
            elif cell == 'o':
                self.maze[pac_tile_y][pac_tile_x] = ' '
                self.score += POWER_DOT_SCORE
                self.pellets_eaten_total_level += 1
                self.pellets_eaten_this_life += 1
                self.activate_power_mode()
                self.pacman.is_eating_dot_timer = 2


    def activate_power_mode(self):
        self.power_mode_active = True
        self.power_mode_timer = POWER_PELLET_TIME
        self.ghost_eat_combo_count = 0 # Reset combo for new power pellet
        sounds['power_pellet_active'].play(-1) # Loop siren

        for ghost in self.ghosts:
            if ghost.mode not in ["dead", "entering_pen", "pen"]:
                ghost.frighten()

    def update_power_mode(self):
        if self.power_mode_active:
            self.power_mode_timer -= 1
            if self.power_mode_timer <= 0:
                self.power_mode_active = False
                if sounds['power_pellet_active'].get_num_channels() > 0:
                    sounds['power_pellet_active'].stop()
                for ghost in self.ghosts:
                    if ghost.is_frightened: # Check is_frightened, not mode == "frightened"
                        ghost.unfrighten(self.ghost_mode)

    def release_ghosts_from_pen(self):
        # Blinky is always out. Pinky is released first (or with Blinky based on some rules).
        # Then Inky, then Clyde, based on dot counts or timers.
        # This is a simplified dot-based release for Pinky, Inky, Clyde.
        if self.ghosts[1].name == "pinky" and self.ghosts[1].mode == "pen" and self.pellets_eaten_this_life >= self.dots_for_ghost_release["pinky"]:
            self.ghosts[1].leave_pen()
        if self.ghosts[2].name == "inky" and self.ghosts[2].mode == "pen" and self.pellets_eaten_this_life >= self.dots_for_ghost_release["inky"]:
            self.ghosts[2].leave_pen()
        if self.ghosts[3].name == "clyde" and self.ghosts[3].mode == "pen" and self.pellets_eaten_this_life >= self.dots_for_ghost_release["clyde"]:
            self.ghosts[3].leave_pen()


    def check_collisions(self):
        if self.state != "PLAYING": return # No collisions if not in active play

        pac_rect = self.pacman.get_rect()
        for ghost in self.ghosts:
            if ghost.mode == "dead" or ghost.mode == "entering_pen": continue

            ghost_rect = ghost.get_rect()
            # More precise collision: distance between centers
            dist_sq = (self.pacman.x - ghost.x)**2 + (self.pacman.y - ghost.y)**2
            # if pac_rect.colliderect(ghost_rect): # Original check
            if dist_sq < (self.pacman.radius + ghost.radius - 4)**2: # Collision radius sum (minus a bit for feel)
                if ghost.is_frightened:
                    ghost.get_eaten()
                    score_to_add = GHOST_EAT_SCORES[min(self.ghost_eat_combo_count, len(GHOST_EAT_SCORES)-1)]
                    self.score += score_to_add
                    # TODO: Display floating score for eaten ghost
                    self.ghost_eat_combo_count += 1
                    sounds['ghost_eaten'].play()
                    self.state = "GHOST_EAT_PAUSE"
                    self.state_timer = int(0.5 * FPS) # Pause for 0.5 seconds
                    # During pause, only Pacman and eaten ghost are hidden/shown as score. Other ghosts freeze.
                else:
                    self.state = "DYING"
                    self.state_timer = PACMAN_DEATH_TIME
                    sounds['death'].play()
                    if sounds['power_pellet_active'].get_num_channels() > 0:
                        sounds['power_pellet_active'].stop()
                    return # Stop further updates this frame (Pacman died)


    def check_level_completion(self):
        if self.pellets_eaten_total_level >= self.total_pellets_in_maze:
            self.state = "LEVEL_CLEAR"
            self.state_timer = 2 * FPS # Show clear screen for 2 seconds (maze flashes)
            if sounds['power_pellet_active'].get_num_channels() > 0:
                sounds['power_pellet_active'].stop()


    def draw(self):
        self.screen.fill(BLACK)
        flash_maze_on_clear = self.state == "LEVEL_CLEAR" and (self.state_timer // (FPS // 8)) % 2 == 0 # Faster flash
        
        if not flash_maze_on_clear:
            self.draw_maze()

        if self.state == "PLAYING" or self.state == "READY" or self.state == "GHOST_EAT_PAUSE" or self.state == "LEVEL_CLEAR":
            if not (self.state == "DYING"): # Don't draw normal Pacman if dying
                 self.pacman.draw(self.screen)

            for ghost in self.ghosts:
                if self.state == "GHOST_EAT_PAUSE" and ghost.mode == "dead" and ghost.just_eaten_timer > 0:
                    # TODO: Draw score text instead of ghost eyes
                    pass # For now, dead ghost (eyes) will draw
                elif self.state == "DYING": # Ghosts freeze and are hidden in original during Pac death anim start
                    pass # Don't draw ghosts if Pacman is in death animation
                else:
                    ghost.draw(self.screen)

        elif self.state == "DYING":
            self.pacman.draw_death_animation(self.screen, self.state_timer, PACMAN_DEATH_TIME)
            # Ghosts are typically hidden during Pac-Man's death animation

        self.draw_ui()
        pygame.display.flip()

    def draw_maze(self):
        for r, row_list in enumerate(self.maze):
            for c, cell_char in enumerate(row_list):
                rect = pygame.Rect(c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                if cell_char == '#':
                    pygame.draw.rect(self.screen, WALL_BLUE, rect)
                elif cell_char == '.':
                    pygame.draw.circle(self.screen, YELLOW, rect.center, 3) # Smaller dots
                elif cell_char == 'o':
                    if self.power_pellet_blink_timer < 10 or self.state == "LEVEL_CLEAR": # Blinking effect, or always show if level clear and not flashing maze
                        pygame.draw.circle(self.screen, YELLOW, rect.center, 7) # Larger power pellets
                elif cell_char == '-': # Ghost door
                    pygame.draw.line(self.screen, PINK, (rect.left, rect.centery), (rect.right, rect.centery), 3)


    def draw_ui(self):
        score_surf = self.font.render(f"{self.score:02d}", True, WHITE)
        self.screen.blit(score_surf, (60, SCREEN_HEIGHT - 90))
        self.screen.blit(self.small_font.render("1UP", True, WHITE), (60, SCREEN_HEIGHT - 100 - TILE_SIZE //2 ))


        hiscore_surf = self.font.render(f"{self.high_score:02d}", True, WHITE)
        self.screen.blit(hiscore_surf, (SCREEN_WIDTH // 2 - hiscore_surf.get_width()//2, SCREEN_HEIGHT - 90))
        self.screen.blit(self.small_font.render("HIGH SCORE", True, WHITE), (SCREEN_WIDTH // 2 - self.small_font.size("HIGH SCORE")[0]//2 , SCREEN_HEIGHT - 100 - TILE_SIZE //2))


        # Draw lives
        for i in range(self.lives -1): # -1 because current life isn't shown as icon
            life_rect = pygame.Rect(20 + i * (TILE_SIZE + 5), SCREEN_HEIGHT - 40, TILE_SIZE, TILE_SIZE)
            pygame.draw.circle(self.screen, YELLOW, life_rect.center, TILE_SIZE // 2 - 2)
            pygame.draw.polygon(self.screen, BLACK, [(life_rect.centerx + TILE_SIZE // 4, life_rect.centery),
                                                     (life_rect.centerx - TILE_SIZE//6, life_rect.centery - TILE_SIZE//6),
                                                     (life_rect.centerx - TILE_SIZE//6, life_rect.centery + TILE_SIZE//6)])


        if self.state == "READY" and self.state_timer > 0 : # Only show READY if timer is active
            ready_surf = self.font.render("READY!", True, YELLOW)
            self.screen.blit(ready_surf, ready_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + TILE_SIZE*2))) # Adjusted Y pos
        elif self.state == "GAME_OVER":
            go_surf = self.font.render("GAME OVER", True, RED)
            self.screen.blit(go_surf, go_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + TILE_SIZE)))
        elif self.state == "START_SCREEN":
            title_surf = self.font.render("PAC-MAN by Cat-san", True, YELLOW)
            prompt_surf = self.small_font.render("Press ENTER to Start", True, WHITE)
            self.screen.blit(title_surf, title_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3)))
            self.screen.blit(prompt_surf, prompt_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))


    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if self.state == "START_SCREEN" and event.key == pygame.K_RETURN:
                        self.start_new_game()
                    if self.state == "GAME_OVER" and event.key == pygame.K_RETURN: # Allow return to title from game over
                         self.state = "START_SCREEN"


            # Input handling should depend on state
            if self.state == "PLAYING":
                 self.handle_input()

            self.update() # Main game logic update
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()

class Pacman:
    def __init__(self, game, start_tile_pos): # start_tile_pos (col, row)
        self.game = game
        self.start_pixel_pos = (start_tile_pos[0] * TILE_SIZE, start_tile_pos[1] * TILE_SIZE)
        self.radius = TILE_SIZE // 2 - 2
        self.mouth_animation_timer = 0
        self.mouth_open_angle = math.pi / 6
        self.is_eating_dot_timer = 0
        self.reset_position()


    def reset_position(self):
        self.x, self.y = self.start_pixel_pos
        self.current_dir = (0, 0) # No initial movement until first input or READY state ends
        self.next_dir_buffered = (-1, 0) # Arcade PacMan often starts facing left and can move
        self.logical_facing_dir = (-1,0)
        self.speed = 1.8 # Base speed

    def set_next_direction(self, direction):
        self.next_dir_buffered = direction

    def get_tile(self):
        return int(round(self.x / TILE_SIZE)), int(round(self.y / TILE_SIZE))

    def get_pixel_pos_snapped_to_tile_center(self, tile_col, tile_row):
        return tile_col * TILE_SIZE + TILE_SIZE // 2, tile_row * TILE_SIZE + TILE_SIZE // 2

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)

    def update(self, maze):
        current_speed = self.speed
        if self.is_eating_dot_timer > 0:
            current_speed *= 0.9
            self.is_eating_dot_timer -=1

        # --- Direction Change Logic ---
        current_tile_col_f = self.x / TILE_SIZE
        current_tile_row_f = self.y / TILE_SIZE
        
        # Use a small tolerance for checking alignment with tile center
        tolerance = current_speed * 0.6 # Must be significantly centered

        can_evaluate_turn = False
        if self.current_dir[0] != 0: # Moving horizontally
            if abs(current_tile_row_f - (round(current_tile_row_f))) < tolerance / TILE_SIZE:
                can_evaluate_turn = True
                self.y = round(current_tile_row_f) * TILE_SIZE # Snap Y
        elif self.current_dir[1] != 0: # Moving vertically
            if abs(current_tile_col_f - (round(current_tile_col_f))) < tolerance / TILE_SIZE:
                can_evaluate_turn = True
                self.x = round(current_tile_col_f) * TILE_SIZE # Snap X
        elif self.current_dir == (0,0): # Stopped, can always try to turn
             can_evaluate_turn = True


        if can_evaluate_turn and self.next_dir_buffered:
            # Current tile based on snapped position for decision
            # For turn decision, use integer tile coords Pac-Man is on or mostly on
            # Snapping ensures Pac-Man is at a grid line for the perpendicular axis
            eval_tile_col = int(round(self.x / TILE_SIZE))
            eval_tile_row = int(round(self.y / TILE_SIZE))

            next_potential_tile_col = eval_tile_col + self.next_dir_buffered[0]
            next_potential_tile_row = eval_tile_row + self.next_dir_buffered[1]

            # Check if the turn is into a non-wall tile
            if 0 <= next_potential_tile_row < MAZE_ROWS and 0 <= next_potential_tile_col < MAZE_COLS and \
               self.game.maze[next_potential_tile_row][next_potential_tile_col] != '#':
                # If turning 90 degrees, ensure snapped to current tile center for axis of previous movement
                if self.current_dir[0] != 0 and self.next_dir_buffered[1] != 0: # Was horizontal, turning vertical
                    self.x = eval_tile_col * TILE_SIZE + (TILE_SIZE // 2 if self.current_dir[0] > 0 else -TILE_SIZE // 2) if self.x % TILE_SIZE != 0 else self.x # Snap to center/edge based on float pos
                    self.x = round(self.x / TILE_SIZE) * TILE_SIZE # Ensure on grid line
                elif self.current_dir[1] != 0 and self.next_dir_buffered[0] != 0: # Was vertical, turning horizontal
                    self.y = round(self.y / TILE_SIZE) * TILE_SIZE # Ensure on grid line


                self.current_dir = self.next_dir_buffered
                self.logical_facing_dir = self.current_dir
                self.next_dir_buffered = None # Clear buffer
            # If trying to reverse, allow it immediately if path is clear (already handled by general logic)
            elif self.next_dir_buffered[0] == -self.current_dir[0] and self.next_dir_buffered[1] == -self.current_dir[1]:
                self.current_dir = self.next_dir_buffered
                self.logical_facing_dir = self.current_dir
                self.next_dir_buffered = None


        # --- Movement and Collision ---
        # Predict next position based on integer tile logic first
        current_map_tile_col, current_map_tile_row = int(round(self.x / TILE_SIZE)), int(round(self.y / TILE_SIZE))

        # Wall collision before moving
        next_tile_c_check = current_map_tile_col + self.current_dir[0]
        next_tile_r_check = current_map_tile_row + self.current_dir[1]

        # Check if attempting to move into a wall from current tile
        # This requires Pacman to be fairly aligned with the grid for this check to be simple
        is_aligned_x = abs(self.x - (current_map_tile_col * TILE_SIZE + TILE_SIZE // 2)) < current_speed
        is_aligned_y = abs(self.y - (current_map_tile_row * TILE_SIZE + TILE_SIZE // 2)) < current_speed

        blocked = False
        if self.current_dir[0] != 0 and is_aligned_y: # Moving horizontally, aligned on Y
            if not (0 <= next_tile_c_check < MAZE_COLS and 0 <= current_map_tile_row < MAZE_ROWS and \
                    self.game.maze[current_map_tile_row][next_tile_c_check] != '#'):
                blocked = True
                self.x = current_map_tile_col * TILE_SIZE + TILE_SIZE // 2 # Snap to center
        elif self.current_dir[1] != 0 and is_aligned_x: # Moving vertically, aligned on X
             if not (0 <= current_map_tile_col < MAZE_COLS and 0 <= next_tile_r_check < MAZE_ROWS and \
                    self.game.maze[next_tile_r_check][current_map_tile_col] != '#'):
                blocked = True
                self.y = current_map_tile_row * TILE_SIZE + TILE_SIZE // 2 # Snap to center
        
        if blocked:
            self.current_dir = (0,0) # Stop


        # Actual move
        self.x += self.current_dir[0] * current_speed
        self.y += self.current_dir[1] * current_speed


        # Tunnel wrapping (needs to use pixel positions more directly)
        # Pacman's center x,y for tunnel. Radius is for drawing and precise collision.
        tunnel_row_idx = 14 # As per maze_layout comment
        if int(round(self.y / TILE_SIZE)) == tunnel_row_idx: # If Pacman is on the tunnel row
            if self.x < 0 - TILE_SIZE // 2 : # Allow full body to exit before wrap
                self.x = SCREEN_WIDTH + TILE_SIZE // 2
            elif self.x > SCREEN_WIDTH + TILE_SIZE // 2:
                self.x = 0 - TILE_SIZE // 2


        # Mouth animation (symmetric open/close)
        self.mouth_animation_timer = (self.mouth_animation_timer + 1) % 16
        if self.current_dir != (0,0):
            if self.mouth_animation_timer < 8 :
                self.mouth_open_angle = math.pi / (4 + self.mouth_animation_timer / 2.0)
            else:
                self.mouth_open_angle = math.pi / (4 + (15 - self.mouth_animation_timer) / 2.0)
            self.mouth_open_angle = max(0.05, min(math.pi/3.5, self.mouth_open_angle)) # Ensure mouth is always a bit open
        else:
            self.mouth_open_angle = math.pi / 6


    def draw(self, screen):
        pos = (int(self.x), int(self.y))
        base_angle_rad = 0
        if self.logical_facing_dir == (1,0): base_angle_rad = 0
        elif self.logical_facing_dir == (-1,0): base_angle_rad = math.pi
        elif self.logical_facing_dir == (0,-1): base_angle_rad = math.pi * 1.5
        elif self.logical_facing_dir == (0,1): base_angle_rad = math.pi * 0.5

        if self.mouth_open_angle > 0.01 :
            poly_points = [pos]
            num_steps = 20
            
            body_arc_start_angle = base_angle_rad + self.mouth_open_angle / 2.0
            body_arc_end_angle = base_angle_rad - self.mouth_open_angle / 2.0
            
            if body_arc_end_angle <= body_arc_start_angle: # Ensure end angle is CCW greater for sweep
                body_arc_end_angle += 2 * math.pi

            total_body_sweep = body_arc_end_angle - body_arc_start_angle # Should be (2*pi - mouth_open_angle)
            
            angle_step = total_body_sweep / num_steps

            for i in range(num_steps + 1):
                current_angle = body_arc_start_angle + i * angle_step
                poly_points.append((pos[0] + self.radius * math.cos(current_angle),
                                    pos[1] + self.radius * math.sin(current_angle)))
            
            if len(poly_points) > 2:
                pygame.draw.polygon(screen, YELLOW, poly_points)
        else: # Mouth closed or nearly closed, draw full circle
            pygame.draw.circle(screen, YELLOW, pos, self.radius)

    def draw_death_animation(self, screen, timer_value, max_timer_value):
        center = (int(self.x), int(self.y))
        progress = (max_timer_value - timer_value) / max_timer_value

        num_segments = 12
        base_radius = self.radius
        for i in range(num_segments):
            angle = (2 * math.pi / num_segments) * i + (progress * math.pi * 3) # Faster rotation
            current_radius_factor = (1 - progress**0.5) # Shrink slower at start, faster at end
            if current_radius_factor < 0: current_radius_factor = 0

            outer_dist = base_radius * 0.5 + progress * base_radius * 4 # Start closer, expand further

            p_center_offset_x = center[0] + (outer_dist * 0.3 * math.sin(progress * math.pi)) * math.cos(angle + math.pi/2) # Swirl effect
            p_center_offset_y = center[1] + (outer_dist * 0.3 * math.sin(progress * math.pi)) * math.sin(angle + math.pi/2)

            p1 = (p_center_offset_x + outer_dist * math.cos(angle),
                  p_center_offset_y + outer_dist * math.sin(angle))
            # Make segments thinner as they fly out
            segment_thickness_angle = 0.1 * (1-progress) + 0.02
            p2 = (p_center_offset_x + (outer_dist + base_radius * current_radius_factor * 0.8) * math.cos(angle + segment_thickness_angle),
                  p_center_offset_y + (outer_dist + base_radius * current_radius_factor * 0.8) * math.sin(angle + segment_thickness_angle))
            p3 = (p_center_offset_x + (outer_dist + base_radius * current_radius_factor * 0.8) * math.cos(angle - segment_thickness_angle),
                  p_center_offset_y + (outer_dist + base_radius * current_radius_factor * 0.8) * math.sin(angle - segment_thickness_angle))

            if current_radius_factor > 0.02 :
                pygame.draw.polygon(screen, YELLOW, [p1, p2, p3])


class Ghost:
    def __init__(self, game, name, start_tile_pos, color, scatter_target_tile):
        self.game = game
        self.name = name
        # start_tile_pos is (col, row), potentially with .5 for centering
        self.start_pixel_pos = (start_tile_pos[0] * TILE_SIZE, start_tile_pos[1] * TILE_SIZE)
        self.color = color
        self.original_color = color
        self.scatter_target_pixel = (scatter_target_tile[0] * TILE_SIZE + TILE_SIZE//2,
                                     scatter_target_tile[1] * TILE_SIZE + TILE_SIZE//2)
        self.radius = TILE_SIZE // 2 - 2 # Visual radius
        self.collision_radius = TILE_SIZE // 2 - 3 # Slightly smaller for collision
        self.is_frightened = False
        self.frightened_timer = 0
        self.speed = 1.4
        self.reverse_direction_at_next_intersection = False
        self.just_eaten_timer = 0 # For showing score briefly
        self.original_mode_before_fright = "scatter" # Default
        self.reset_to_start_or_pen(game.ghost_mode)


    def reset_to_start_or_pen(self, current_game_ghost_mode):
        self.x, self.y = self.start_pixel_pos # This uses the .5 values correctly
        self.current_target_pixel = self.scatter_target_pixel
        self.is_frightened = False
        self.frightened_timer = 0
        self.color = self.original_color
        self.speed = 1.4 # Reset speed
        self.just_eaten_timer = 0

        if self.name == "blinky":
            self.mode = current_game_ghost_mode
            # Place Blinky slightly above and centered on ghost house exit tile for pixel pos
            self.x = GHOST_HOME_EXIT_TILE[0] * TILE_SIZE + TILE_SIZE // 2
            self.y = GHOST_HOME_EXIT_TILE[1] * TILE_SIZE + TILE_SIZE // 2
            self.current_dir = (-1, 0) # Starts moving left usually
        else:
            self.mode = "pen"
            # For Pinky, Inky, Clyde, start_pixel_pos already accounts for their specific x coords like 13.5, 11.5, 15.5
            # Bobbing direction
            if self.y < GHOST_PEN_CENTER_TILE[1]*TILE_SIZE: self.current_dir = (0,1) # Move down if above center
            else: self.current_dir = (0,-1) # Move up if below or at center

        self.reverse_direction_at_next_intersection = False


    def get_tile(self): # Tile ghost is currently on (center based)
        return int(round(self.x / TILE_SIZE)), int(round(self.y / TILE_SIZE))

    def get_pixel_pos_snapped_to_tile_center(self, tile_col, tile_row):
        return tile_col * TILE_SIZE + TILE_SIZE // 2, tile_row * TILE_SIZE + TILE_SIZE // 2

    def get_rect(self): # Used for rough collision, more precise can use distance
        return pygame.Rect(self.x - self.collision_radius, self.y - self.collision_radius, self.collision_radius * 2, self.collision_radius * 2)

    def set_mode(self, new_mode):
        if self.mode != new_mode: # Avoid unnecessary reversals if mode is re-asserted
            self.mode = new_mode
            if self.mode in ["scatter", "chase"] and not self.is_frightened: # Don't reverse if just became unfrightened
                 self.reverse_direction_at_next_intersection = True


    def frighten(self):
        if self.mode not in ["dead", "entering_pen"]: # Cannot frighten dead/returning ghosts
            if not self.is_frightened : # Only trigger reversal if not already frightened
                 self.reverse_direction_at_next_intersection = True
            self.is_frightened = True
            self.frightened_timer = POWER_PELLET_TIME
            self.original_mode_before_fright = self.mode # Store current mode to return to (e.g. scatter/chase)
            self.mode = "frightened" # Actual mode for logic
            self.speed = 0.9


    def unfrighten(self, game_ghost_mode_now):
        self.is_frightened = False
        self.frightened_timer = 0
        self.color = self.original_color
        self.speed = 1.4
        # Revert to the mode the game is currently in (scatter/chase), not necessarily what it was before fright
        self.mode = game_ghost_mode_now
        self.reverse_direction_at_next_intersection = True


    def get_eaten(self):
        self.mode = "dead"
        self.is_frightened = False # No longer blue/flashing
        self.frightened_timer = 0
        # self.color = self.original_color # Eyes will be drawn based on "dead" mode
        self.speed = 2.8 # Faster when returning
        self.current_target_pixel = self.get_pixel_pos_snapped_to_tile_center(GHOST_HOME_EXIT_TILE[0], GHOST_HOME_EXIT_TILE[1] -1) # Target tile above door
        self.just_eaten_timer = 1 * FPS # Show score for 1 sec


    def leave_pen(self):
        if self.mode == "pen":
            self.mode = "leaving_pen"
            # Target the spot just outside (above) the ghost house door
            self.current_target_pixel = self.get_pixel_pos_snapped_to_tile_center(GHOST_HOME_EXIT_TILE[0], GHOST_HOME_EXIT_TILE[1])
            # Ensure x is centered on the door tile x for vertical exit
            self.x = GHOST_HOME_EXIT_TILE[0] * TILE_SIZE + TILE_SIZE // 2


    def _is_tile_valid_for_move(self, tile_col, tile_row, maze, can_use_door=False):
        if not (0 <= tile_row < MAZE_ROWS and 0 <= tile_col < MAZE_COLS):
            return False
        cell = maze[tile_row][tile_col]
        if cell == '#': return False
        if cell == '-' and not can_use_door: return False
        return True


    def update_target_pixel(self, pacman, blinky_ref, game_ghost_mode, maze):
        pac_tile_col, pac_tile_row = pacman.get_tile()
        # Use Pacman's pixel position for more fluid targeting by ghosts
        pac_pixel_x, pac_pixel_y = pacman.x, pacman.y
        pac_facing_dir = pacman.logical_facing_dir


        if self.mode == "dead":
            # Target: above door -> inside pen (specific spot for this ghost name)
            current_tile_col, current_tile_row = self.get_tile()
            target_door_entry_tile_y = GHOST_HOME_EXIT_TILE[1] # Row of the door itself
            
            if current_tile_row == target_door_entry_tile_y and \
               abs(self.x - (GHOST_HOME_EXIT_TILE[0] * TILE_SIZE + TILE_SIZE//2)) < self.speed:
                self.mode = "entering_pen"
                # Target specific X,Y inside pen based on original start pos
                self.current_target_pixel = (GHOST_START_POSITIONS[self.name][0] * TILE_SIZE,
                                             GHOST_START_POSITIONS[self.name][1] * TILE_SIZE)
            else: # Still heading for above door / door entry
                 self.current_target_pixel = self.get_pixel_pos_snapped_to_tile_center(GHOST_HOME_EXIT_TILE[0], target_door_entry_tile_y)
            return

        if self.mode == "entering_pen":
            self.current_target_pixel = (GHOST_START_POSITIONS[self.name][0] * TILE_SIZE,
                                         GHOST_START_POSITIONS[self.name][1] * TILE_SIZE)
            if abs(self.x - self.current_target_pixel[0]) < self.speed and \
               abs(self.y - self.current_target_pixel[1]) < self.speed:
                self.reset_to_start_or_pen(game_ghost_mode) # Resets mode to "pen" or active if blinky
            return


        if self.mode == "frightened": # Random movement handled in main update logic
            return

        if self.mode == "scatter":
            self.current_target_pixel = self.scatter_target_pixel
            return

        # CHASE MODE TARGETING (using pixel positions of Pacman for more accuracy)
        target_pixel_x, target_pixel_y = pac_pixel_x, pac_pixel_y

        if self.name == "blinky":
            pass # Default target is Pac-Man's pixel position

        elif self.name == "pinky":
            offset = 4 * TILE_SIZE
            target_pixel_x += pac_facing_dir[0] * offset
            target_pixel_y += pac_facing_dir[1] * offset
            if pac_facing_dir == (0, -1): # Up (original bug: also go left)
                target_pixel_x -= offset # Bug: 4 tiles left

        elif self.name == "inky":
            blinky_pixel_x, blinky_pixel_y = blinky_ref.x, blinky_ref.y
            offset_val = 2 * TILE_SIZE
            
            pac_offset_target_x = pac_pixel_x + pac_facing_dir[0] * offset_val
            pac_offset_target_y = pac_pixel_y + pac_facing_dir[1] * offset_val
            if pac_facing_dir == (0,-1): # Pac-Man moving UP (original bug for offset)
                pac_offset_target_x -= offset_val # Offset also includes the left move for Inky's reference point

            vec_x = pac_offset_target_x - blinky_pixel_x
            vec_y = pac_offset_target_y - blinky_pixel_y
            target_pixel_x = pac_offset_target_x + vec_x
            target_pixel_y = pac_offset_target_y + vec_y

        elif self.name == "clyde":
            my_pixel_x, my_pixel_y = self.x, self.y
            dist_sq_to_pac = (my_pixel_x - pac_pixel_x)**2 + (my_pixel_y - pac_pixel_y)**2
            if dist_sq_to_pac < (8 * TILE_SIZE)**2:
                self.current_target_pixel = self.scatter_target_pixel # Use scatter target directly (pixel)
                return
            # Else, target Pac-Man (default behavior using pac_pixel_x,y)

        self.current_target_pixel = (target_pixel_x, target_pixel_y)


    def update(self, pacman, blinky_ref, maze, game_ghost_mode):
        if self.just_eaten_timer > 0:
            self.just_eaten_timer -=1
            # Ghost is "invisible" (eyes only) and pathing back
            # Movement logic for "dead" mode will handle this

        current_tile_col, current_tile_row = self.get_tile() # Integer tile ghost is on
        center_of_current_tile_x, center_of_current_tile_y = self.get_pixel_pos_snapped_to_tile_center(current_tile_col, current_tile_row)

        # Update target pixel based on mode and Pac-Man's position
        if self.mode not in ["pen", "leaving_pen"]:
            self.update_target_pixel(pacman, blinky_ref, game_ghost_mode, maze)

        if self.is_frightened:
            self.frightened_timer -= 1
            if self.frightened_timer <= 0:
                self.unfrighten(game_ghost_mode) # Pass current game mode

        # --- Decision Making at Intersections (Tile Centers) ---
        # Ghosts make decisions when their center is very close to a tile's center.
        is_centered_on_tile = abs(self.x - center_of_current_tile_x) < self.speed * 0.5 and \
                               abs(self.y - center_of_current_tile_y) < self.speed * 0.5

        if is_centered_on_tile:
            self.x = center_of_current_tile_x # Snap to center for clean decisions
            self.y = center_of_current_tile_y

            if self.reverse_direction_at_next_intersection and self.current_dir != (0,0) :
                potential_rev_dir = (-self.current_dir[0], -self.current_dir[1])
                next_rev_col = current_tile_col + potential_rev_dir[0]
                next_rev_row = current_tile_row + potential_rev_dir[1]
                can_use_door_for_rev = self.mode in ["leaving_pen", "entering_pen", "dead"]

                if self._is_tile_valid_for_move(next_rev_col, next_rev_row, maze, can_use_door_for_rev):
                    self.current_dir = potential_rev_dir
                # If cannot reverse, will proceed to normal direction picking below
                self.reverse_direction_at_next_intersection = False # Consumed the reversal attempt


            if self.mode == "pen":
                pen_top_y_boundary = GHOST_START_POSITIONS[self.name][1] * TILE_SIZE - TILE_SIZE * 0.75
                pen_bottom_y_boundary = GHOST_START_POSITIONS[self.name][1] * TILE_SIZE + TILE_SIZE * 0.75
                if self.y <= pen_top_y_boundary : self.current_dir = (0, 1)
                elif self.y >= pen_bottom_y_boundary: self.current_dir = (0, -1)
                self.x = GHOST_START_POSITIONS[self.name][0] * TILE_SIZE # Maintain X position

            elif self.mode == "leaving_pen":
                # Move towards GHOST_HOME_EXIT_TILE (pixel target)
                # Logic: if X is not aligned with door center, move X. Then move Y up.
                door_center_x = GHOST_HOME_EXIT_TILE[0] * TILE_SIZE + TILE_SIZE//2
                door_center_y = GHOST_HOME_EXIT_TILE[1] * TILE_SIZE + TILE_SIZE//2

                if abs(self.x - door_center_x) > self.speed * 0.5 : # If not aligned on X with door center
                    self.current_dir = (1 if door_center_x > self.x else -1, 0)
                elif abs(self.y - door_center_y) > self.speed * 0.5 : # X is aligned, now move Y up to door center
                    self.current_dir = (0, -1 if door_center_y < self.y else 1) # Move towards door_center_y
                    self.x = door_center_x # Ensure X stays snapped
                else: # Reached center of tile outside door
                    self.mode = game_ghost_mode # Transition to current game mode
                    self.set_mode(game_ghost_mode) # This might trigger reverse
                    if self.current_dir == (0,1) or self.current_dir == (0,0): # If somehow facing down or stopped, exit left
                        self.current_dir = (-1,0)


            elif self.mode == "frightened":
                possible_dirs = []
                for dx, dy in [(0,-1), (0,1), (-1,0), (1,0)]: # U, D, L, R
                    if (dx,dy) == (-self.current_dir[0], -self.current_dir[1]) and self.current_dir != (0,0): continue # No reversing
                    next_c, next_r = current_tile_col + dx, current_tile_row + dy
                    if self._is_tile_valid_for_move(next_c, next_r, maze, False):
                        possible_dirs.append((dx,dy))
                if possible_dirs:
                    self.current_dir = random.choice(possible_dirs)
                elif self.current_dir != (0,0) and self._is_tile_valid_for_move(current_tile_col - self.current_dir[0], current_tile_row - self.current_dir[1], maze, False):
                    self.current_dir = (-self.current_dir[0], -self.current_dir[1]) # Only option is reverse

            elif self.mode in ["scatter", "chase", "dead", "entering_pen"]:
                best_dir = None
                min_dist_sq = float('inf')
                ordered_dirs_for_choice = [(0, -1), (-1, 0), (0, 1), (1, 0)] # U, L, D, R preference

                no_up_turn_zones = [(12,11), (15,11), (12,23), (15,23)] # Tiles before tunnels etc.
                can_turn_up = not ( (current_tile_col, current_tile_row) in no_up_turn_zones and \
                                   self.mode not in ["dead", "frightened", "entering_pen", "leaving_pen"] ) # Allow up for these critical modes

                potential_choices = []
                for dx, dy in ordered_dirs_for_choice:
                    if (dx, dy) == (-self.current_dir[0], -self.current_dir[1]) and self.current_dir!=(0,0):
                        continue
                    if not can_turn_up and (dx, dy) == (0, -1):
                        continue

                    next_tile_c, next_tile_r = current_tile_col + dx, current_tile_row + dy
                    can_use_door = self.mode in ["leaving_pen", "entering_pen", "dead"]

                    if self._is_tile_valid_for_move(next_tile_c, next_tile_r, maze, can_use_door):
                        target_x_px, target_y_px = self.current_target_pixel
                        # Project ghost's next tile center for distance calculation
                        next_potential_center_x = (next_tile_c * TILE_SIZE) + TILE_SIZE // 2
                        next_potential_center_y = (next_tile_r * TILE_SIZE) + TILE_SIZE // 2
                        dist_sq = (next_potential_center_x - target_x_px)**2 + (next_potential_center_y - target_y_px)**2
                        potential_choices.append(((dx,dy), dist_sq))
                
                if potential_choices:
                    potential_choices.sort(key=lambda item: item[1]) # Sort by distance
                    self.current_dir = potential_choices[0][0]
                elif self.current_dir != (0,0) : # No valid choices, try to reverse if possible
                    rev_dir = (-self.current_dir[0], -self.current_dir[1])
                    next_tile_c, next_tile_r = current_tile_col + rev_dir[0], current_tile_row + rev_dir[1]
                    can_use_door = self.mode in ["leaving_pen", "entering_pen", "dead"]
                    if self._is_tile_valid_for_move(next_tile_c, next_tile_r, maze, can_use_door):
                        self.current_dir = rev_dir
                    # else: self.current_dir = (0,0) # Stuck

        # --- Actual Movement ---
        current_speed_eff = self.speed
        on_tunnel_row = (current_tile_row == 14 and 0 < current_tile_col < MAZE_COLS -1)
        if on_tunnel_row and self.mode not in ["dead", "entering_pen"]:
            current_speed_eff *= 0.6

        self.x += self.current_dir[0] * current_speed_eff
        self.y += self.current_dir[1] * current_speed_eff

        # Tunnel wrapping based on pixel position
        if current_tile_row == 14: # Tunnel row index
            if self.x < 0 - TILE_SIZE // 2: self.x = SCREEN_WIDTH + TILE_SIZE // 2 -1 # -1 to avoid immediate re-wrap
            elif self.x > SCREEN_WIDTH + TILE_SIZE // 2: self.x = 0 - TILE_SIZE // 2 +1


    def draw(self, screen):
        if self.just_eaten_timer > 0 and self.mode == "dead":
             # Draw score text at ghost's last position before eaten (or current if moving)
            score_val = GHOST_EAT_SCORES[min(self.game.ghost_eat_combo_count -1, len(GHOST_EAT_SCORES)-1)] # -1 as combo was incremented
            score_surf = self.game.small_font.render(str(score_val), True, CYAN if score_val < 1000 else PINK)
            screen.blit(score_surf, (self.x - score_surf.get_width()//2, self.y - score_surf.get_height()//2 - 5))
            return # Don't draw eyes if showing score

        pos = (int(self.x), int(self.y))
        body_rect_width = self.radius * 2
        body_rect_height = self.radius * 1.5
        head_radius = self.radius

        draw_color = self.original_color
        if self.mode == "dead":
            draw_color = None # Eyes only
        elif self.is_frightened:
            fright_duration_left_ratio = self.frightened_timer / POWER_PELLET_TIME
            if fright_duration_left_ratio < 0.3 and (self.frightened_timer // (FPS // 6)) % 2 == 0 : # Flash faster near end
                draw_color = WHITE
            else:
                draw_color = BLUE_GHOST_FRIGHTENED

        if draw_color:
            pygame.draw.circle(screen, draw_color, (pos[0], pos[1] - head_radius // 3), head_radius)
            body_top_y = pos[1] - head_radius // 3
            pygame.draw.rect(screen, draw_color, (pos[0] - head_radius, body_top_y, head_radius * 2, body_rect_height))

            num_waves = 3
            wave_width = (head_radius * 2) / num_waves
            wave_base_y = body_top_y + body_rect_height
            wave_amplitude = head_radius / 2.5
            
            # Use sin wave for smoother bottom
            for i in range(int(head_radius * 2)): # Iterate pixels across width
                x_local = i - head_radius # from -radius to +radius
                # make 3 full waves across the width:
                angle_wave = (x_local / (head_radius*2)) * math.pi * num_waves * 2
                y_offset = math.sin(angle_wave + (pygame.time.get_ticks()/200.0)) * wave_amplitude/3 # Gentle bobbing
                
                pygame.draw.line(screen, draw_color,
                                 (pos[0] + x_local, wave_base_y + y_offset),
                                 (pos[0] + x_local, wave_base_y + wave_amplitude + y_offset) )


        # Eyes
        eye_outer_radius = self.radius * 0.40
        eye_pupil_radius = self.radius * 0.20
        eye_offset_x = self.radius * 0.45
        eye_base_y = pos[1] - self.radius * 0.4

        pupil_shift_x, pupil_shift_y = 0, 0
        pupil_offset_amount = eye_outer_radius * 0.4

        if self.mode == "dead":
             pupil_shift_y = 0; pupil_shift_x = 0 # Center pupils when dead (classic look) or look towards target
             # Or look towards target:
             if self.current_target_pixel:
                dx = self.current_target_pixel[0] - self.x
                dy = self.current_target_pixel[1] - self.y
                dist = math.hypot(dx, dy)
                if dist > 0:
                    pupil_shift_x = (dx / dist) * pupil_offset_amount
                    pupil_shift_y = (dy / dist) * pupil_offset_amount

        elif self.current_dir != (0,0):
            pupil_shift_x = self.current_dir[0] * pupil_offset_amount
            pupil_shift_y = self.current_dir[1] * pupil_offset_amount
        
        # Left Eye
        left_eye_center_x = pos[0] - eye_offset_x
        pygame.draw.circle(screen, WHITE, (int(left_eye_center_x), int(eye_base_y)), int(eye_outer_radius))
        pupil_color = BLACK
        if self.is_frightened: pupil_color = BLUE_GHOST_FRIGHTENED # Dark blue pupils on light blue body
        if self.mode == "dead": pupil_color = (100,100,200) # Light blue pupils for eyes when dead

        pygame.draw.circle(screen, pupil_color,
                           (int(left_eye_center_x + pupil_shift_x), int(eye_base_y + pupil_shift_y)), int(eye_pupil_radius))

        # Right Eye
        right_eye_center_x = pos[0] + eye_offset_x
        pygame.draw.circle(screen, WHITE, (int(right_eye_center_x), int(eye_base_y)), int(eye_outer_radius))
        pygame.draw.circle(screen, pupil_color,
                           (int(right_eye_center_x + pupil_shift_x), int(eye_base_y + pupil_shift_y)), int(eye_pupil_radius))


if __name__ == "__main__":
    game = Game()
    game.run()
