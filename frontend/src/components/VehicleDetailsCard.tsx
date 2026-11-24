import React, { useState } from 'react';
import { ArrowLeft, Car } from 'lucide-react';
import { EvidenceUrls, VehicleAttributes } from '../types';

interface VehicleDetailsCardProps {
  vehicle: VehicleAttributes;
  evidenceUrls: EvidenceUrls;
  onAnalyzeAnother: () => void;
}

export const VehicleDetailsCard: React.FC<VehicleDetailsCardProps> = ({
  vehicle,
  evidenceUrls,
  onAnalyzeAnother,
}) => {
  const [viewMode, setViewMode] = useState<'crop' | 'strip'>('crop');

  const hasStrip = Boolean(evidenceUrls.vehicle_contact_strip_url);
  const activeImage =
    viewMode === 'strip' && hasStrip
      ? evidenceUrls.vehicle_contact_strip_url
      : evidenceUrls.best_vehicle_crop_url;

  return (
    <div className="bg-white rounded-2xl border border-[#E6E6E2] p-6 shadow-[0_1px_3px_rgba(0,0,0,0.02)] flex flex-col gap-5">
      <div className="flex items-center justify-between border-b border-[#E6E6E2] pb-3">
        <div className="flex items-center gap-2">
          <Car className="w-4 h-4 text-[#111111]" />
          <span className="text-xs font-semibold uppercase tracking-wider text-[#111111]">
            Vehicle Classification &amp; Observations
          </span>
        </div>

        {vehicle.provider_used && (
          <span className="font-mono text-[11px] text-[#6B6B66]">
            {vehicle.provider_used}
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
        {/* Left: Live Media Frame with Multi-Angle Strip Toggle */}
        <div className="flex flex-col gap-2">
          <div className="relative aspect-video w-full rounded-xl bg-[#111111] overflow-hidden border border-[#E6E6E2] flex items-center justify-center">
            {activeImage ? (
              <img
                src={activeImage}
                alt="Target Vehicle Evidence"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="text-xs font-mono text-[#6B6B66]">No visual crop available</div>
            )}

            {/* Pill Toggle for Contact Strip */}
            {hasStrip && (
              <div className="absolute top-2.5 right-2.5 flex items-center bg-black/75 backdrop-blur-md rounded-lg p-0.5 border border-white/10 text-xs font-mono">
                <button
                  onClick={() => setViewMode('crop')}
                  className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
                    viewMode === 'crop'
                      ? 'bg-white text-[#111111]'
                      : 'text-white/70 hover:text-white'
                  }`}
                >
                  Primary Crop
                </button>
                <button
                  onClick={() => setViewMode('strip')}
                  className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
                    viewMode === 'strip'
                      ? 'bg-white text-[#111111]'
                      : 'text-white/70 hover:text-white'
                  }`}
                >
                  1×3 Multi-Angle
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Right: 2x2 Attribute Grid + Return Button */}
        <div className="flex flex-col justify-between gap-5 h-full">
          <div className="grid grid-cols-2 gap-3 font-mono text-xs tabular-nums">
            <div className="p-3 rounded-xl bg-[#F7F7F5] border border-[#E6E6E2] flex flex-col gap-1">
              <span className="text-[10px] text-[#6B6B66] uppercase">Body Type</span>
              <span className="font-semibold text-xs text-[#111111]">{vehicle.body_type || 'Unknown'}</span>
              <span className="text-[10px] text-[#047857]">
                {Math.round((vehicle.body_type_conf || 0) * 100)}%
              </span>
            </div>

            <div className="p-3 rounded-xl bg-[#F7F7F5] border border-[#E6E6E2] flex flex-col gap-1">
              <span className="text-[10px] text-[#6B6B66] uppercase">Colour</span>
              <span className="font-semibold text-xs text-[#111111]">{vehicle.colour || 'Unknown'}</span>
              <span className="text-[10px] text-[#047857]">
                {Math.round((vehicle.colour_conf || 0) * 100)}%
              </span>
            </div>

            <div className="p-3 rounded-xl bg-[#F7F7F5] border border-[#E6E6E2] flex flex-col gap-1">
              <span className="text-[10px] text-[#6B6B66] uppercase">Make</span>
              <span className="font-semibold text-xs text-[#111111]">{vehicle.make || 'Unknown'}</span>
              <span className="text-[10px] text-[#047857]">
                {Math.round((vehicle.make_conf || 0) * 100)}%
              </span>
            </div>

            <div className="p-3 rounded-xl bg-[#F7F7F5] border border-[#E6E6E2] flex flex-col gap-1">
              <span className="text-[10px] text-[#6B6B66] uppercase">Model</span>
              <span className="font-semibold text-xs text-[#111111]">{vehicle.model || 'Unknown'}</span>
              <span className="text-[10px] text-[#047857]">
                {Math.round((vehicle.model_conf || 0) * 100)}%
              </span>
            </div>
          </div>

          <div className="pt-2 border-t border-[#E6E6E2] flex justify-end">
            <button
              onClick={onAnalyzeAnother}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-[#E6E6E2] bg-[#F1F1EE] hover:bg-[#EAEAE6] text-xs font-medium text-[#111111] transition-colors focus:outline-none"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>← Analyze Another Vehicle</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
