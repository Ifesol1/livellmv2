import { useRef, useEffect, useState } from "react"
import { useFrame, useThree } from "@react-three/fiber"
import { PointerLockControls } from "@react-three/drei"
import * as THREE from "three"

const SPEED = 5
const SPRINT_MULTIPLIER = 1.8
const JUMP_FORCE = 6
const GRAVITY = 3

export default function PlayerController({ onMove }: { onMove?: (pos: THREE.Vector3) => void }) {
  const { camera } = useThree()
  const [moveForward, setMoveForward] = useState(false)
  const [moveBackward, setMoveBackward] = useState(false)
  const [moveLeft, setMoveLeft] = useState(false)
  const [moveRight, setMoveRight] = useState(false)
  
  // Physics state
  const canJump = useRef(false)
  const velocity = useRef(new THREE.Vector3())
  const direction = useRef(new THREE.Vector3())
  const isLocked = useRef(false)

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      switch (event.code) {
        case "ArrowUp":
        case "KeyW":
          setMoveForward(true)
          break
        case "ArrowLeft":
        case "KeyA":
          setMoveLeft(true)
          break
        case "ArrowDown":
        case "KeyS":
          setMoveBackward(true)
          break
        case "ArrowRight":
        case "KeyD":
          setMoveRight(true)
          break
        case "Space":
          if (canJump.current) {
            velocity.current.y = JUMP_FORCE
            canJump.current = false
          }
          break
      }
    }

    const onKeyUp = (event: KeyboardEvent) => {
      switch (event.code) {
        case "ArrowUp":
        case "KeyW":
          setMoveForward(false)
          break
        case "ArrowLeft":
        case "KeyA":
          setMoveLeft(false)
          break
        case "ArrowDown":
        case "KeyS":
          setMoveBackward(false)
          break
        case "ArrowRight":
        case "KeyD":
          setMoveRight(false)
          break
      }
    }

    document.addEventListener("keydown", onKeyDown)
    document.addEventListener("keyup", onKeyUp)

    return () => {
      document.removeEventListener("keydown", onKeyDown)
      document.removeEventListener("keyup", onKeyUp)
    }
  }, [])

  useFrame((state, delta) => {
    if (!isLocked.current) return

    // Friction
    velocity.current.x -= velocity.current.x * 10.0 * delta
    velocity.current.z -= velocity.current.z * 10.0 * delta
    // Low-gravity vertical motion
    velocity.current.y -= GRAVITY * delta 
    
    // Direction calculation
    direction.current.z = Number(moveForward) - Number(moveBackward)
    direction.current.x = Number(moveRight) - Number(moveLeft)
    direction.current.normalize() // this ensures consistent movements in all directions

    if (moveForward || moveBackward) velocity.current.z -= direction.current.z * 400.0 * delta
    if (moveLeft || moveRight) velocity.current.x -= direction.current.x * 400.0 * delta

    // Apply movement
    const moveX = -velocity.current.x * delta
    const moveZ = -velocity.current.z * delta
    
    // Very simple collision (don't walk into the center reactor at 0,0)
    // Camera position is the player position
    const nextX = camera.position.x + moveX
    const nextZ = camera.position.z + moveZ
    const distFromCenter = Math.sqrt(nextX * nextX + nextZ * nextZ)
    
    // Reactor radius ~2.5, buffer to 3.5
    if (distFromCenter > 3.5) {
        // Safe to move
        // PointerLockControls `moveRight` and `moveForward` handle camera relative movement
        // But we are manually calculating velocity. 
        // Actually PointerLockControls has built-in methods but we are building a custom physics-ish feel
        // Let's rely on the library's movement methods if we weren't doing custom velocity
        // But we want momentum.
        
        // Correct way with Three.js PointerLock is usually:
        // controls.moveRight(velocity.x * delta)
        // controls.moveForward(velocity.z * delta)
        // But we need the ref to the controls.
        
        // Since we don't have direct access to the `moveRight/Forward` imperative API easily inside the loop 
        // without a ref to the component instance, let's just move the camera directly relative to its quaternion.
        
        // Better approach for R3F:
        // Use the manual movement vector applied to camera
        
        const speed = 10 * delta
        if (moveForward) camera.translateZ(-speed)
        if (moveBackward) camera.translateZ(speed)
        if (moveLeft) camera.translateX(-speed)
        if (moveRight) camera.translateX(speed)
    } else {
        // Collision pushback
         const angle = Math.atan2(camera.position.z, camera.position.x)
         camera.position.x = Math.cos(angle) * 3.6
         camera.position.z = Math.sin(angle) * 3.6
    }
    
    // Vertical movement and floor clamp
    camera.position.y += velocity.current.y * delta
    if (camera.position.y <= 2) {
      camera.position.y = 2
      velocity.current.y = 0
      canJump.current = true
    }

    if (onMove) onMove(camera.position)

  })

  return (
    <PointerLockControls 
        onLock={() => isLocked.current = true} 
        onUnlock={() => isLocked.current = false}
    />
  )
}
