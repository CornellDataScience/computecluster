import json
import subprocess
import time
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

prev_idle_time, prev_total_time = 0, 0

def get_system_stats():
        """Collect all system statistics"""
        stats = {
            'timestamp': int(time.time()),
            'temperature': get_cpu_temperature(),
            'cpu_usage': get_cpu_usage(),
            'memory': get_memory_usage(),
            'disk': get_disk_usage(),
            'load_average': get_load_average(),
            'uptime': get_uptime(),
            'processes': get_running_processes()
        }
        return stats

def get_cpu_temperature():
        """Get CPU temperature from thermal zone /sys/class/hwmon/hwmon4/temp1_input"""
        if (sys.argv[1] == "compute1"):
            path = '/sys/class/hwmon/hwmon0/temp2_input'
        elif (sys.argv[1] == "compute2"):
            path = '/sys/class/thermal/thermal_zone0/temp'
        else:
            path = '/sys/class/thermal/thermal_zone0/temp'
        try:
            with open(path, 'r') as f:
                temp = int(f.read().strip()) / 1000.0
            return round(temp, 1)
        except Exception as e:
            print(f"Error getting CPU temperature: {e}")
            return None
    
def get_cpu_usage():
    global prev_idle_time, prev_total_time
    
    try:
        with open('/proc/stat') as f:
            fields = [float(column) for column in f.readline().strip().split()[1:]]
        idle_time, total_time = fields[3], sum(fields)
        
        delta_idle_time, delta_total_time = idle_time - prev_idle_time, total_time - prev_total_time
        prev_idle_time, prev_total_time = idle_time, total_time
        
        if delta_total_time == 0:
            return 0.0

        usage = 100.0 * (1.0 - delta_idle_time / delta_total_time)
        return round(usage, 1)
    except Exception as e:
        print(f"Error getting CPU usage: {e}")
        return None

def get_memory_usage():
    try:
        meminfo = {}
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                key, value = line.split(':', 1)
                meminfo[key.strip()] = int(value.strip().split()[0])

        total = meminfo['MemTotal']
        available = meminfo['MemAvailable']
        used = total - available
        usage_percent = (used / total) * 100
        
        return {
            'total_kb': total,
            'used_kb': used,
            'available_kb': available,
            'usage_percent': round(usage_percent, 1)
        }
    except Exception as e:
        print(f"Error getting memory usage: {e}")
        return None

def get_disk_usage():
    try:
        stat = os.statvfs('/')
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bavail * stat.f_frsize
        used = total - free
        usage_percent = (used / total) * 100

        return {
            'total_gb': round(total / (1024**3), 1),
            'used_gb': round(used / (1024**3), 1),
            'available_gb': round(free / (1024**3), 1),
            'usage_percent': round(usage_percent, 1)
        }
    except Exception as e:
        print(f"Error getting disk usage: {e}")
        return None

def get_load_average():
    """Get system load average"""
    try:
        loads = os.getloadavg()
        return {'1min': loads[0], '5min': loads[1], '15min': loads[2]}
    except Exception as e:
        print(f"Error getting load average: {e}")
        return None

def get_uptime():
    """Get system uptime"""
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.read().split()[0])
        
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        
        return f"{days}d {hours}h {minutes}m"
    except Exception as e:
        print(f"Error getting uptime: {e}")
        return None

def get_running_processes():
    """Get top 5 running processes by CPU"""
    try:
        result = subprocess.run(
            ['ps', 'aux'], 
            capture_output=True, text=True, check=True, timeout=5
        )
        lines = result.stdout.strip().split('\n')[1:]
        
        processes = []
        for line in lines:
            if not line.strip(): 
                continue
            parts = line.split(None, 10)
            if len(parts) >= 11:
                try:
                    cpu_usage = float(parts[2])
                    processes.append({
                        'user': parts[0],
                        'pid': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'command': parts[10][:60] if len(parts) > 10 else 'N/A'
                    })
                except ValueError:
                    continue 
        
        processes.sort(key=lambda x: float(x['cpu']), reverse=True)
        return processes[:5]
        
    except Exception as e:
        print(f"Error getting processes: {e}")
        return []
    
def main():
    name = sys.argv[1] + '.json'
    while (True):
        values = get_system_stats()
        with open(name, 'w') as f:
            json.dump(values, f, indent = 2)
        time.sleep(3)
main()