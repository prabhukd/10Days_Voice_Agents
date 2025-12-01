'use client';

import { AnimatePresence, motion } from 'framer-motion';
import { type ReceivedChatMessage } from '@livekit/components-react';
import { ShimmerText } from '@/components/livekit/shimmer-text';
import { cn } from '@/lib/utils';

const MotionMessage = motion.p;

const VIEW_MOTION_PROPS = {
  initial: { opacity: 0 },
  animate: {
    opacity: 1,
    transition: {
      ease: 'easeIn',
      duration: 0.5,
      delay: 0.8,
    },
  },
  exit: {
    opacity: 0,
    transition: {
      ease: 'easeIn',
      duration: 0.5,
    },
  },
};

interface PreConnectMessageProps {
  messages?: ReceivedChatMessage[];
  className?: string;
}

export function PreConnectMessage({
  className,
  messages = [],
}: PreConnectMessageProps) {
  return (
    <AnimatePresence>
      {messages.length === 0 && (
        <MotionMessage
          {...VIEW_MOTION_PROPS}
          className={cn('pointer-events-none text-center', className)}
        >
          <ShimmerText className="text-sm font-semibold">
            Host is listening, show your talent
          </ShimmerText>
        </MotionMessage>
      )}
    </AnimatePresence>
  );
}
