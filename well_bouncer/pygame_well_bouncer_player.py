# Copyright 2019 Brian Quinlan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Use pygame to visualize well_bouncer_game.Game."""

import pygame

import well_bouncer_game

_screen = None

_PLAY_AREA_WIDTH = 200
_PLAY_AREA_HEIGHT = 800
_BORDER_WIDTH = 10
_PLAY_AREA_SCALE = 8


def game_to_screen_point(x, y):
    x = float(x) * _PLAY_AREA_SCALE + _BORDER_WIDTH
    y = float(y) * _PLAY_AREA_SCALE
    return int(x), _PLAY_AREA_HEIGHT + _BORDER_WIDTH - int(y)


def scale_to_screen(p):
    p = float(p) * _PLAY_AREA_SCALE
    return int(p)


class InteractiveMoveMaker(well_bouncer_game.MoveMaker):
    def make_move(self, state) -> well_bouncer_game.Direction:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    return well_bouncer_game.Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    return well_bouncer_game.Direction.RIGHT
        return well_bouncer_game.Direction.CENTER


def _initialize_screen():
    global _screen

    pygame.init()
    pygame.font.init()
    size = 220, 820

    _screen = pygame.display.set_mode(size, pygame.SRCALPHA)
    pygame.key.set_repeat(1, 50)


def play_game(title, game, move_maker: well_bouncer_game.MoveMaker):
    if _screen is None:
        _initialize_screen()

    pygame.display.set_caption(title)
    score_font = pygame.font.SysFont("Consolas", 60)
    clock = pygame.time.Clock()
    score_width = 0
    while not game.done:
        direction = move_maker.make_move(game.state)
        for event in pygame.event.get():
            pass

        game.move(direction)
        _screen.fill((0, 0, 0))

        move_distribution = move_maker.move_probabilities(game.state)
        if move_distribution is not None:
            for i in range(well_bouncer_game.Game.NUM_ACTIONS):
                probability = move_distribution.get(i, 0.0)
                c = pygame.Color(0)
                c.hsva = (60 * i, 100, int(100 * probability), 20)
                pygame.draw.rect(_screen, c,
                                 pygame.Rect(66 * i, 0, 66 * i + 66, 8))

        score_surface = score_font.render(str(int(game.score)), False,
                                          (255, 255, 0))
        score_surface.set_alpha(64)

        new_score_width = score_surface.get_rect().width
        if new_score_width > score_width:
            score_width = new_score_width + 10
        _screen.blit(score_surface,
                     ((_screen.get_rect().width - score_width) // 2, 10))
        pygame.draw.circle(
            _screen,
            (255, 0, 0),
            game_to_screen_point(game.ball_x, game.ball_y),
            scale_to_screen(game.ball_radius),
        )
        pygame.draw.circle(
            _screen,
            (0, 0, 255),
            game_to_screen_point(game.paddle_x, game.paddle_y),
            scale_to_screen(game.paddle_radius),
        )
        pygame.draw.rect(
            _screen,
            (255, 255, 255),
            pygame.Rect(
                0,
                # Start off screen so that there is no top border
                -_BORDER_WIDTH,
                _PLAY_AREA_WIDTH + 2 * _BORDER_WIDTH,
                # 3 * _BORDER_WIDTH because the rectangle started off screen.
                _PLAY_AREA_HEIGHT + 3 * _BORDER_WIDTH),
            _BORDER_WIDTH)
        pygame.display.flip()
        clock.tick(60)
