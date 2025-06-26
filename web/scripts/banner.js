#!/usr/bin/env node

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  brightGreen: '\x1b[92m',
  dim: '\x1b[2m',
  white: '\x1b[37m',
  brightYellow: '\x1b[93m',
  brightBlue: '\x1b[94m',
  brightCyan: '\x1b[96m'
};

const banner = `
${colors.brightGreen}
${colors.brightGreen} ██████╗ ██████╗ ████████╗
${colors.brightGreen}██╔════╝ ██╔══██╗╚══██╔══╝
${colors.brightGreen}██║  ███╗██████╔╝   ██║   
${colors.brightGreen}██║   ██║██╔═══╝    ██║   
${colors.brightGreen}╚██████╔╝██║        ██║   
${colors.brightGreen}╚═════╝ ╚═╝        ╚═╝   

${colors.brightGreen} █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗████████╗██╗ ██████╗███████╗
${colors.brightGreen}██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝╚══██╔══╝██║██╔════╝██╔════╝
${colors.brightGreen}███████║██╔██╗ ██║███████║██║   ╚████╔╝    ██║   ██║██║     ███████╗
${colors.brightGreen}██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝     ██║   ██║██║     ╚════██║
${colors.brightGreen}██║  ██║██║ ╚████║██║  ██║███████╗██║      ██║   ██║╚██████╗███████║
${colors.brightGreen}╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝      ╚═╝   ╚═╝ ╚═════╝╚══════╝
${colors.reset}

${colors.brightCyan}→${colors.reset} ${colors.white}Frontend: ${colors.brightBlue}http://localhost:3000${colors.reset}
${colors.brightCyan}→${colors.reset} ${colors.white}Backend:  ${colors.brightBlue}http://localhost:8000${colors.reset}

${colors.dim}Ready to analyze! ${colors.brightYellow}●${colors.reset}
`;

console.log(banner); 