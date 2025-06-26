"use client";

import { useEffect } from 'react';

export default function CursorTrail() {
  useEffect(() => {
    let trailElements: HTMLElement[] = [];
    let customCursor: HTMLElement | null = null;
    let lastTrailTime = 0;
    const trailDelay = 100; // Milliseconds between trail elements

    // Create custom cursor
    function createCustomCursor() {
      customCursor = document.createElement('div');
      customCursor.className = 'custom-cursor';
      document.body.appendChild(customCursor);
    }

    // Create trail element
    function createTrailElement(x: number, y: number) {
      const trail = document.createElement('div');
      trail.className = 'cursor-trail';
      trail.style.left = (x - 3) + 'px';
      trail.style.top = (y - 3) + 'px';
      
      // Add some randomness to the trail
      const randomOffset = (Math.random() - 0.5) * 6;
      trail.style.transform = `translate(${randomOffset}px, ${randomOffset}px)`;
      
      document.body.appendChild(trail);
      trailElements.push(trail);

      // Remove trail element after animation
      setTimeout(() => {
        if (trail.parentNode) {
          trail.parentNode.removeChild(trail);
        }
        const index = trailElements.indexOf(trail);
        if (index > -1) {
          trailElements.splice(index, 1);
        }
      }, 600);
    }

    // Mouse move handler
    function handleMouseMove(e: MouseEvent) {
      const currentTime = Date.now();
      
      // Update custom cursor position
      if (customCursor) {
        customCursor.style.left = (e.clientX - 3) + 'px';
        customCursor.style.top = (e.clientY - 3) + 'px';
      }

      // Create trail elements with delay
      if (currentTime - lastTrailTime > trailDelay) {
        createTrailElement(e.clientX, e.clientY);
        lastTrailTime = currentTime;
      }
    }

    // Hide custom cursor when mouse leaves window
    function handleMouseLeave() {
      if (customCursor) {
        customCursor.style.opacity = '0';
      }
    }

    // Show custom cursor when mouse enters window
    function handleMouseEnter() {
      if (customCursor) {
        customCursor.style.opacity = '1';
      }
    }

    // Initialize cursor effects
    createCustomCursor();
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseleave', handleMouseLeave);
    document.addEventListener('mouseenter', handleMouseEnter);

    // Cleanup function
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseleave', handleMouseLeave);
      document.removeEventListener('mouseenter', handleMouseEnter);
      
      // Remove custom cursor
      if (customCursor && customCursor.parentNode) {
        customCursor.parentNode.removeChild(customCursor);
      }
      
      // Remove any remaining trail elements
      trailElements.forEach(trail => {
        if (trail.parentNode) {
          trail.parentNode.removeChild(trail);
        }
      });
      trailElements = [];
    };
  }, []);

  return null; // This component doesn't render anything visible
} 