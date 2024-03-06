
      (define (domain ghost)

          (:requirements
              :typing
              :negative-preconditions
          )

          (:types
              invaders cells
          )

          (:predicates
              (cell ?p)

              ;Pacman's cell location
              (at-ghost ?loc - cells)

              ;Invaders cell location
              (at-invader ?i - invaders ?loc - cells)

              ;Capsule cell location
              (at-capsule ?loc - cells)

              ;Connects cells
              (connected ?from ?to - cells)

          )

          ; move ghost to invader
          (:action move
              :parameters (?from ?to - cells)
              :precondition (and 
                  (at-ghost ?from)
                  (connected ?from ?to)
              )
              :effect (and
                          (at-ghost ?to)
                          (not (at-ghost ?from))       
                      )
          )

          ; Eat invader
          (:action eat-invader
              :parameters (?loc - cells ?i - invaders)
              :precondition (and 
                              (at-ghost ?loc)
                              (at-invader ?i ?loc)
                            )
              :effect (and
                          (not (at-invader ?i ?loc))
                      )
          )
      )
      