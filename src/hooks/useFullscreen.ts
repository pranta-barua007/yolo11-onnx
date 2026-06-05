"use client";

import { useEffect, useState, useCallback } from "react";

interface VendorDocument extends Document {
  webkitFullscreenElement?: Element;
  mozFullScreenElement?: Element;
  msFullscreenElement?: Element;
  webkitExitFullscreen?: () => Promise<void>;
  mozCancelFullScreen?: () => Promise<void>;
  msExitFullscreen?: () => Promise<void>;
}

interface VendorHTMLElement extends HTMLElement {
  webkitRequestFullscreen?: () => Promise<void>;
  mozRequestFullScreen?: () => Promise<void>;
  msRequestFullscreen?: () => Promise<void>;
}

/**
 * Reusable hook to toggle and track browser fullscreen state of a specific DOM element.
 * Handles vendor prefixes and ESC/gesture exits automatically.
 */
export function useFullscreen(elementRef: React.RefObject<HTMLElement | null>) {
  const [isFullscreen, setIsFullscreen] = useState(false);

  const checkFullscreen = useCallback(() => {
    const doc = document as VendorDocument;
    const isFull = !!(
      doc.fullscreenElement ||
      doc.webkitFullscreenElement ||
      doc.mozFullScreenElement ||
      doc.msFullscreenElement
    );
    setIsFullscreen(isFull);
  }, []);

  const toggleFullscreen = useCallback(async () => {
    const element = elementRef.current;
    if (!element) return;

    try {
      const doc = document as VendorDocument;
      const el = element as VendorHTMLElement;

      const currentFullscreenElement =
        doc.fullscreenElement ||
        doc.webkitFullscreenElement ||
        doc.mozFullScreenElement ||
        doc.msFullscreenElement;

      if (!currentFullscreenElement) {
        // Enter fullscreen
        if (el.requestFullscreen) {
          await el.requestFullscreen();
        } else if (el.webkitRequestFullscreen) {
          await el.webkitRequestFullscreen();
        } else if (el.mozRequestFullScreen) {
          await el.mozRequestFullScreen();
        } else if (el.msRequestFullscreen) {
          await el.msRequestFullscreen();
        }
      } else {
        // Exit fullscreen
        if (doc.exitFullscreen) {
          await doc.exitFullscreen();
        } else if (doc.webkitExitFullscreen) {
          await doc.webkitExitFullscreen();
        } else if (doc.mozCancelFullScreen) {
          await doc.mozCancelFullScreen();
        } else if (doc.msExitFullscreen) {
          await doc.msExitFullscreen();
        }
      }
    } catch (err) {
      console.error("[useFullscreen] Error toggling fullscreen:", err);
    }
  }, [elementRef]);

  useEffect(() => {
    const handleEvents = ["fullscreenchange", "webkitfullscreenchange", "mozfullscreenchange", "MSFullscreenChange"];
    
    handleEvents.forEach(evt => {
      document.addEventListener(evt, checkFullscreen);
    });

    return () => {
      handleEvents.forEach(evt => {
        document.removeEventListener(evt, checkFullscreen);
      });
    };
  }, [checkFullscreen]);

  return { isFullscreen, toggleFullscreen };
}
