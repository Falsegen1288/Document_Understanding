/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Search, Bell, Settings, CornerDownRight, ShieldCheck } from 'lucide-react';
import { ActiveView } from '../types';
import { USER_PROFILES } from '../data';

interface TopNavBarProps {
  activeView: ActiveView;
  setActiveView: (view: ActiveView) => void;
  systemStatus: string;
}

export const TopNavBar: React.FC<TopNavBarProps> = ({
  activeView,
  setActiveView,
  systemStatus,
}) => {
  const [searchQuery, setSearchQuery] = useState('');

  // Map view to profile pic for pixel-perfect visual reference matching prompt screenshots
  const getProfileImage = () => {
    switch (activeView) {
      case 'vision':
        return USER_PROFILES.vision;
      case 'table':
        return USER_PROFILES.table;
      case 'ocr':
        return USER_PROFILES.ocr;
      case 'layout':
        return USER_PROFILES.layout;
      default:
        return USER_PROFILES.inspector;
    }
  };

  return (
    <header className="fixed top-0 left-0 right-0 h-16 bg-background border-b border-outline-variant flex justify-between items-center px-6 z-50 select-none">
      <div className="flex items-center gap-8">
        <span 
          onClick={() => setActiveView('landing')}
          className="font-headline text-2xl font-black text-primary tracking-tight cursor-pointer hover:opacity-85 select-none"
          id="brand-title"
        >
          Obsidian Precision
        </span>
        <nav className="hidden md:flex gap-6 items-center">
          <button 
            onClick={() => setActiveView('landing')}
            className={`font-sans text-xs tracking-wider font-semibold uppercase hover:text-primary transition-colors cursor-pointer select-none pb-1 ${
              activeView === 'landing' ? 'text-primary border-b-2 border-primary' : 'text-on-surface-variant'
            }`}
          >
            Overview
          </button>
          <button 
            onClick={() => setActiveView('ingestion')}
            className={`font-sans text-xs tracking-wider font-semibold uppercase hover:text-primary transition-colors cursor-pointer select-none pb-1 ${
              activeView === 'ingestion' ? 'text-primary border-b-2 border-primary' : 'text-on-surface-variant'
            }`}
          >
            Ingestion Pipeline
          </button>
          <button 
            onClick={() => setActiveView('layout')}
            className={`font-sans text-xs tracking-wider font-semibold uppercase hover:text-primary transition-colors cursor-pointer select-none pb-1 ${
              activeView === 'layout' || activeView === 'ocr' || activeView === 'table' || activeView === 'vision' ? 'text-primary border-b-2 border-primary' : 'text-on-surface-variant'
            }`}
          >
            Engines
          </button>
          <span className="text-on-surface-variant/40 text-xs">|</span>
          <span className="text-[10px] custom-font-mono text-secondary tracking-widest uppercase flex items-center gap-1.5 bg-secondary-container/10 px-2 py-0.5 rounded border border-secondary-container/20">
            <span className="w-1.5 h-1.5 rounded-full bg-secondary pulse-running" />
            {systemStatus}
          </span>
        </nav>
      </div>

      <div className="flex items-center gap-4">
        <div className="relative hidden sm:block">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-outline w-4 h-4" />
          <input 
            type="text" 
            placeholder="Search param hashes..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                alert(`Search query "${searchQuery}" submitted: Scanning localized database for matching SHA-256 blocks...`);
              }
            }}
            className="w-64 bg-surface-container-lowest border border-outline-variant rounded-lg pl-10 pr-4 py-1.5 text-xs text-on-surface font-sans placeholder-outline focus:ring-1 focus:ring-primary focus:outline-none transition-all"
          />
        </div>

        {/* Action Controls */}
        <button 
          onClick={() => alert('Diagnostic Queue status update: \n- 0 jobs in critical waiting state.\n- Node Alpha pipeline stream normal.\n- Thermal cores operating within safe temperature limits.')}
          title="Notification Stack"
          className="p-2 text-on-surface-variant hover:text-primary transition-colors active:scale-95 duration-100 cursor-pointer"
        >
          <Bell className="w-5 h-5 animate-pulse-slow" />
        </button>
        <button 
          onClick={() => alert('Obsidian Precision global cluster options loaded. Current active engine is Obsidian-Pipeline v4.2.0 ML.')}
          title="System Settings"
          className="p-2 text-on-surface-variant hover:text-primary transition-colors active:scale-95 duration-100 cursor-pointer"
        >
          <Settings className="w-5 h-5" />
        </button>

        {/* Multi-Context User Avatar */}
        <div 
          className="w-8 h-8 rounded-full overflow-hidden border border-outline-variant select-none cursor-pointer hover:border-primary/50 transition-colors"
          onClick={() => alert('Access level: DATA_OPERATOR_ADMIN (Security cleared).')}
        >
          <img 
            ref={null}
            alt="Data Operator headshot" 
            className="w-full h-full object-cover" 
            src={getProfileImage()}
          />
        </div>
      </div>
    </header>
  );
};
