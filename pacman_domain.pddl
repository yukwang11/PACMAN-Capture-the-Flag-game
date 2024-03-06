
    (define (domain pacman)

      (:requirements
          :typing
          :negative-preconditions
      )

      (:types
          foods cells
      )

      (:predicates
          (cell ?p)

          ;Pacman's cell location
          (at-pacman ?loc - cells)

          ;Food cell location
          (at-food ?f - foods ?loc - cells)

          ;Ghost location
          (at-ghost ?loc - cells)

          ;Capsule cell location
          (at-capsule ?loc - cells)

          ;Connects cells
          (connected ?from ?to - cells)

          ;Capsule has been eaten 
          (non-capsule)

          (carrying-food)

          (go-die)

          (die)
      )

      ;Pacman can move if the
      ;    - Pacman is at current location
      ;    - cells are connected
      ; move pacman to location with no ghost
      (:action move
          :parameters (?from ?to - cells)
          :precondition (and
              (not (at-ghost ?to))
              (at-pacman ?from)
              (connected ?from ?to)
          )
          :effect (and
                      (at-pacman ?to)
                      (not (at-pacman ?from))
                  )
      )

      ;When this action is executed, the pacman go to next location which may have ghost
      (:action move-non-limit
          :parameters (?from ?to - cells)
          :precondition (and
              (at-pacman ?from)
              (connected ?from ?to)
              (go-die)
          )
          :effect (and
                      ;; add
                      (at-pacman ?to)
                      ;; del
                      (not (at-pacman ?from))
                  )
      )

      ;Pacman eats foods
      (:action eat-food
          :parameters (?loc - cells ?f - foods)
          :precondition (and
                          (at-pacman ?loc)
                          (at-food ?f ?loc)
                        )
          :effect (and
                      (carrying-food)
                      (not (at-food ?f ?loc))
                  )
      )

      ;Pacman eats capsule
      (:action eat-capsule
          :parameters (?loc - cells)
          :precondition (and
                          (at-pacman ?loc)
                          (at-capsule ?loc)
                        )
          :effect (and
                      (non-capsule)
                      (not (at-capsule ?loc))
                  )
      )

      ;Pacman moves after eat capsule
      (:action move-with-capsule
          :parameters (?from ?to - cells)
          :precondition (and
              (at-pacman ?from)
              (connected ?from ?to)
              (non-capsule)
          )
          :effect (and
                      (at-pacman ?to)
                      (not (at-pacman ?from))
                  )
      )

      ;Pacman meet the ghost and go to die
      (:action death
          :parameters (?loc - cells)
          :precondition (and
                          (at-pacman ?loc)
                          (at-ghost ?loc)
                        )
          :effect (and
                      (die)
                      (not(carrying-food))
                  )
      )
    )
    