"use client";

import React, { useRef, useState, useMemo, useEffect } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Text, Float, Sparkles, PerspectiveCamera, Cylinder, Box, Torus, Html, useTexture } from "@react-three/drei";
import * as THREE from "three";
import PlayerController from "./PlayerController";

interface FactoryProps {
  status: "nominal" | "warning" | "critical";
  pressure: number;
  onInteract?: (station: string) => void;
}

// --- SUB-COMPONENTS ---

function Floor() {
  return (
    <group>
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
        <planeGeometry args={[100, 100]} />
        <meshStandardMaterial 
            color="#3a3a3a" 
            roughness={0.6} 
            metalness={0.2} 
        />
        </mesh>
        <gridHelper args={[100, 50, 0x555555, 0x222222]} position={[0, 0.01, 0]} />
    </group>
  );
}

function Wall({ position, rotation, size }: { position: [number, number, number], rotation: [number, number, number], size: [number, number] }) {
    return (
        <mesh position={position} rotation={rotation}>
            <planeGeometry args={size} />
            <meshStandardMaterial color="#666" metalness={0.2} roughness={0.5} />
        </mesh>
    )
}

function Pipe({ position, rotation, length, color = "#555", status }: { position: [number, number, number], rotation: [number, number, number], length: number, color?: string, status?: string }) {
    const isWarning = status === "warning" || status === "critical";
    return (
        <group position={position} rotation={rotation}>
            <Cylinder args={[0.3, 0.3, length, 16]} rotation={[0, 0, Math.PI/2]}>
                <meshStandardMaterial 
                    color={color} 
                    roughness={0.4} 
                    metalness={0.8} 
                    emissive={isWarning ? "#ff4400" : "black"}
                    emissiveIntensity={isWarning ? 0.5 : 0}
                />
            </Cylinder>
            {/* Flanges at ends */}
            <Cylinder args={[0.4, 0.4, 0.2, 16]} position={[length/2, 0, 0]} rotation={[0, 0, Math.PI/2]}>
                <meshStandardMaterial color="#444" roughness={0.4} metalness={0.8} />
            </Cylinder>
            <Cylinder args={[0.4, 0.4, 0.2, 16]} position={[-length/2, 0, 0]} rotation={[0, 0, Math.PI/2]}>
                <meshStandardMaterial color="#444" roughness={0.4} metalness={0.8} />
            </Cylinder>
        </group>
    )
}

function CoolingFan({ speed }: { speed: number }) {
    const ref = useRef<THREE.Group>(null);
    useFrame((state, delta) => {
        if (ref.current) {
            ref.current.rotation.y += delta * speed;
        }
    });

    return (
        <group ref={ref} position={[0, 3.2, 0]}>
            <Box args={[3, 0.1, 0.5]} position={[0, 0, 0]}>
                 <meshStandardMaterial color="#222" />
            </Box>
             <Box args={[0.5, 0.1, 3]} position={[0, 0, 0]}>
                 <meshStandardMaterial color="#222" />
            </Box>
        </group>
    )
}

function Interactable({ position, label, onInteract, isHovered }: { position: [number, number, number], label: string, onInteract: () => void, isHovered: boolean }) {
    const ref = useRef<THREE.Group>(null)
    
    // Float animation
    useFrame((state) => {
        if(ref.current) {
            ref.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 2) * 0.1
        }
    })

    return (
        <group position={position} ref={ref}>
            <Box args={[1, 1, 1]}>
                <meshStandardMaterial 
                    color={isHovered ? "#00ff00" : "#0088ff"} 
                    emissive={isHovered ? "#00ff00" : "#0088ff"}
                    emissiveIntensity={0.5}
                    wireframe
                />
            </Box>
            <Html position={[0, 1.5, 0]} center transform sprite>
                <div className={`px-2 py-1 rounded bg-black/80 border ${isHovered ? 'border-green-500 text-green-500' : 'border-blue-500 text-blue-500'} text-xs font-mono whitespace-nowrap`}>
                    {label}
                </div>
            </Html>
        </group>
    )
}

function ReactorTank({ status, pressure }: { status: string, pressure: number }) {
    const materialRef = useRef<THREE.MeshStandardMaterial>(null);
    
    // Pulse effect
    useFrame((state) => {
        if (materialRef.current) {
            const baseColor = status === "critical" ? new THREE.Color("#ff0000") : 
                             status === "warning" ? new THREE.Color("#ffa500") : 
                             new THREE.Color("#00ff88");
            
            // Pulse emissive intensity based on pressure/status
            const pulseSpeed = status === "critical" ? 10 : 2;
            const intensity = 0.5 + Math.sin(state.clock.elapsedTime * pulseSpeed) * 0.5;
            
            materialRef.current.color.lerp(baseColor, 0.1);
            materialRef.current.emissive = baseColor;
            materialRef.current.emissiveIntensity = status === "critical" ? intensity * 2 : intensity * 0.5;
        }
    });

    return (
        <group position={[0, 2, 0]}>
            {/* Main Tank Body */}
            <Cylinder args={[2, 2, 6, 32]} position={[0, 0, 0]}>
                <meshStandardMaterial color="#333" roughness={0.3} metalness={0.9} />
            </Cylinder>
            
            {/* Top Cap */}
            <mesh position={[0, 3, 0]}>
                <sphereGeometry args={[2, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2]} />
                <meshStandardMaterial color="#333" roughness={0.3} metalness={0.9} />
            </mesh>
            
            {/* Cooling Fan */}
            <CoolingFan speed={status === "critical" ? 10 : status === "warning" ? 5 : 1} />
            
            {/* Bottom Cap */}
             <mesh position={[0, -3, 0]} rotation={[Math.PI, 0, 0]}>
                <sphereGeometry args={[2, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2]} />
                <meshStandardMaterial color="#333" roughness={0.3} metalness={0.9} />
            </mesh>

            {/* Glowing Window/Core */}
            <Cylinder args={[1.5, 1.5, 4, 32]} position={[0, 0, 0]}>
                 <meshStandardMaterial ref={materialRef} transparent opacity={0.9} />
            </Cylinder>

            {/* Cooling Rings */}
            <Torus args={[2.2, 0.15, 16, 32]} position={[0, 1.5, 0]} rotation={[Math.PI/2, 0, 0]}>
                <meshStandardMaterial color="#888" metalness={1} roughness={0.2} />
            </Torus>
            <Torus args={[2.2, 0.15, 16, 32]} position={[0, 0, 0]} rotation={[Math.PI/2, 0, 0]}>
                <meshStandardMaterial color="#888" metalness={1} roughness={0.2} />
            </Torus>
            <Torus args={[2.2, 0.15, 16, 32]} position={[0, -1.5, 0]} rotation={[Math.PI/2, 0, 0]}>
                <meshStandardMaterial color="#888" metalness={1} roughness={0.2} />
            </Torus>
        </group>
    )
}

function SteamParticles({ status }: { status: string }) {
    if (status === "nominal") return null;
    
    const count = status === "critical" ? 100 : 30;
    const color = status === "critical" ? "#ff4400" : "#cccccc";
    const scale = status === "critical" ? 8 : 4;
    const speed = status === "critical" ? 2 : 0.5;

    return (
        <group position={[0, 5, 0]}>
             <Sparkles 
                count={count} 
                scale={[4, 5, 4]} 
                size={scale} 
                speed={speed} 
                opacity={0.5} 
                color={color} 
                noise={0.5}
            />
        </group>
    )
}

function Catwalk() {
    return (
        <group position={[0, 3, 0]}>
            <Box args={[10, 0.2, 2]} position={[0, 0, 0]}>
                <meshStandardMaterial color="#50565b" metalness={0.4} roughness={0.4} />
            </Box>
            <Box args={[10, 0.1, 0.1]} position={[0, 0.8, 1]}>
                <meshStandardMaterial color="#cccccc" metalness={0.3} roughness={0.5} />
            </Box>
            <Box args={[10, 0.1, 0.1]} position={[0, 0.4, 1]}>
                <meshStandardMaterial color="#cccccc" metalness={0.3} roughness={0.5} />
            </Box>
            <Box args={[0.1, 1, 0.1]} position={[-4.5, 0.5, 1]}>
                <meshStandardMaterial color="#cccccc" metalness={0.3} roughness={0.5} />
            </Box>
            <Box args={[0.1, 1, 0.1]} position={[4.5, 0.5, 1]}>
                <meshStandardMaterial color="#cccccc" metalness={0.3} roughness={0.5} />
            </Box>
        </group>
    )
}

function SideTanks() {
    return (
        <group>
            <Cylinder args={[1.2, 1.2, 3, 24]} position={[-8, 1.5, -4]}>
                <meshStandardMaterial color="#2f3b45" metalness={0.7} roughness={0.3} />
            </Cylinder>
            <Cylinder args={[1.2, 1.2, 3, 24]} position={[8, 1.5, -4]}>
                <meshStandardMaterial color="#2f3b45" metalness={0.7} roughness={0.3} />
            </Cylinder>
        </group>
    )
}

function OverheadPipes() {
    return (
        <group position={[0, 5.5, 0]}>
            <Cylinder args={[0.2, 0.2, 20, 16]} rotation={[0, 0, Math.PI / 2]} position={[0, 0, -3]}>
                <meshStandardMaterial color="#999999" metalness={0.8} roughness={0.3} />
            </Cylinder>
            <Cylinder args={[0.2, 0.2, 20, 16]} rotation={[0, 0, Math.PI / 2]} position={[0, 0, 3]}>
                <meshStandardMaterial color="#999999" metalness={0.8} roughness={0.3} />
            </Cylinder>
        </group>
    )
}

function SceneContent({ status, pressure, setHoveredObject }: { status: string, pressure: number, setHoveredObject: (obj: string | null) => void }) {
    const { camera } = useThree()
    
    // Stations definitions
    const stations = [
        { id: 'valve_release', position: [-6, 1, 0] as [number, number, number], label: 'PRESSURE RELEASE' },
        { id: 'coolant_pump', position: [6, 1, 0] as [number, number, number], label: 'COOLANT PUMP' },
        { id: 'main_terminal', position: [0, 1, 6] as [number, number, number], label: 'MAIN TERMINAL' }
    ]

    const [hovered, setHovered] = useState<string | null>(null)

    useFrame(() => {
        // Simple raycasting alternative: Distance check
        let closest = null
        let minDst = 3.0 // Interaction range

        stations.forEach(st => {
            const dist = camera.position.distanceTo(new THREE.Vector3(...st.position))
            if (dist < minDst) {
                closest = st.id
            }
        })

        if (closest !== hovered) {
            setHovered(closest)
            setHoveredObject(closest)
        }
    })

    return (
        <group>
            {/* Global soft, bright light */}
            <hemisphereLight args={["#ffffff", "#666666", 0.8]} />
            <ambientLight intensity={1.0} />

            {/* Ceiling Lights for highlights */}
            <pointLight position={[0, 10, 0]} intensity={1.2} distance={40} />
            <pointLight position={[-10, 10, -10]} intensity={0.8} distance={40} color="#aaf" />
            <pointLight position={[10, 10, 10]} intensity={0.8} distance={40} color="#aaf" />
            
            <spotLight position={[10, 15, 10]} angle={0.35} intensity={1.5} castShadow />
            
            {/* Alarm Light */}
            {status === "critical" && (
                <pointLight position={[0, 8, 0]} intensity={2.5} color="#ff4444" distance={40} decay={2} />
            )}

            <Floor />
            
            {/* Main Reactor */}
            <ReactorTank status={status} pressure={pressure} />
            <SteamParticles status={status} />
            <Catwalk />
            <SideTanks />
            <OverheadPipes />

            {/* Pipes */}
            <Pipe position={[-4, 3, 0]} rotation={[0, 0, 0]} length={4} status={status} />
            <Pipe position={[4, 2, 0]} rotation={[0, 0, 0]} length={4} status={status} />
            
            {/* Stations */}
            {stations.map(st => (
                <Interactable 
                    key={st.id}
                    position={st.position} 
                    label={st.label} 
                    onInteract={() => console.log('Interact', st.id)}
                    isHovered={hovered === st.id}
                />
            ))}

            {/* Walls for boundaries visual */}
            <Wall position={[0, 5, -10]} rotation={[0, 0, 0]} size={[40, 10]} />
            <Wall position={[0, 5, 10]} rotation={[Math.PI, 0, 0]} size={[40, 10]} />
            <Wall position={[-20, 5, 0]} rotation={[0, Math.PI/2, 0]} size={[40, 10]} />
            <Wall position={[20, 5, 0]} rotation={[0, -Math.PI/2, 0]} size={[40, 10]} />

            <PlayerController />
        </group>
    )
}

export default function FactoryScene({ status, pressure }: FactoryProps) {
  const [hoveredObject, setHoveredObject] = useState<string | null>(null)
  
  // Handle 'E' key for interaction
  useEffect(() => {
      const handleKeyPress = (e: KeyboardEvent) => {
          if (e.key.toLowerCase() === 'e' && hoveredObject) {
              console.log("INTERACTED WITH", hoveredObject)
              window.dispatchEvent(new CustomEvent('station-interact', { detail: hoveredObject }))
          }
      }
      window.addEventListener('keydown', handleKeyPress)
      return () => window.removeEventListener('keydown', handleKeyPress)
  }, [hoveredObject])

  return (
    <div className="h-full w-full bg-black overflow-hidden relative select-none">
       {/* Minimal HUD - just crosshair and interaction prompt */}
       <div className="absolute inset-0 z-10 pointer-events-none">
            {/* Crosshair */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-4 h-4 border border-white/30 rounded-full" />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-1 h-1 bg-white/50 rounded-full" />

            {/* Interaction Prompt */}
            {hoveredObject && (
                <div className="absolute top-[60%] left-1/2 -translate-x-1/2 bg-black/70 backdrop-blur text-white px-4 py-2 rounded border border-white/20">
                    <span className="font-bold text-yellow-400 mr-2">[E]</span>
                    INTERACT: {hoveredObject.replace('_', ' ').toUpperCase()}
                </div>
            )}
       </div>

      <Canvas shadows onClick={(e) => (e.target as HTMLElement).requestPointerLock()}>
        <PerspectiveCamera makeDefault position={[0, 2, 10]} fov={75}>
            {/* Player Headlamp */}
            <spotLight position={[0.5, 0, 0]} angle={0.5} penumbra={0.5} intensity={1.5} distance={50} castShadow color="#ffffff" />
        </PerspectiveCamera>
        
        {/* Environment */}
        <color attach="background" args={['#111318']} />
        <fog attach="fog" args={['#111318', 15, 60]} />
        
        <SceneContent status={status} pressure={pressure} setHoveredObject={setHoveredObject} />
      </Canvas>
    </div>
  );
}
