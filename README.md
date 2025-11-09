# Monkeying Around


An Augmented Reality (AR) Body-Tracking Game

Monkeying Around is an interactive AR-style game that uses your computer’s camera to track full-body movement and detect real-time collisions between the player and virtual objects. Built with Python, MediaPipe, and OpenCV, the game combines motion tracking with on-screen gameplay—no controllers required.

Description

Players move in front of the camera to catch falling bananas before they hit the ground. Each successful catch increases the score, but if three bananas fall to the ground, the game ends.

As the game progresses:

- The number of falling bananas increases

- The speed of the bananas accelerates

- The difficulty and movement required by the player increase

Technology Stack

Python: Core programming language for the game logic

OpenCV: Handles video frame capture and image processing

MediaPipe: Enables full-body and hand tracking for real-time interaction

Pygame: Provides audio and basic interface elements

Features

- Real-time body and gesture tracking

- Collision detection between tracked body points and falling objects

- Physics-based banana motion

- Stage progression with increasing difficulty

- Audio feedback for catches and misses

Credits

This project was inspired by and partially based on Full Body Detection with OpenCV and MediaPipe by Ramzan, which served as a guide for implementing MediaPipe’s pose detection features.

Gameplay Summary

Move your body and hands in front of the camera to catch bananas as they fall. If three bananas reach the ground, the game ends. As you advance, more bananas appear and fall faster, testing both your movement and reaction time.


