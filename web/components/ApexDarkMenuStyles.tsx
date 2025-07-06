"use client";
import { useEffect } from 'react';

const CSS = `
.apexcharts-menu{background:#1e293b!important;border:1px solid #334155!important;}
.apexcharts-menu-item{color:#f8fafc!important;}
.apexcharts-menu-item:hover{background:#334155!important;}
.apexcharts-menu svg{fill:#f8fafc!important;}
`;

export default function ApexDarkMenuStyles(){
  useEffect(()=>{
    if(document.getElementById('apexcharts-darkmenu')) return;
    const tag=document.createElement('style');
    tag.id='apexcharts-darkmenu';
    tag.innerHTML=CSS;
    document.head.appendChild(tag);
    return ()=>{tag.remove();};
  },[]);
  return null;
} 