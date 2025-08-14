import requests
import os
from dotenv import load_dotenv

class AbuseipdbCheck:
    """Queries AbuseIPDB API to check if an IP address is malicious"""
    load_dotenv()

    def __init__(self, target, max_age):
        self.target = target
        self.max_age = max_age
        self.api_token = os.getenv('ABUSEIPDB_KEY')
        self.base_url = 'https://api.abuseipdb.com/api/v2'
        self.headers = {'Key': self.api_token}
        self.abuseipdb_categories = {
            3: "Fraud Orders",
            4: "DDoS Attack",
            5: "FTP Brute-Force",
            6: "Ping of Death",
            7: "Phishing",
            8: "Fraud VoIP",
            9: "Open Proxy",
            10: "Web Spam",
            11: "Email Spam",
            12: "Blog Spam",
            13: "VPN IP",
            14: "Port Scan",
            15: "Hacking",
            16: "SQL Injection",
            17: "Spoofing",
            18: "Brute-Force",
            19: "Bad Web Bot",
            20: "Exploited Host",
            21: "Web App Attack",
            22: "SSH",
            23: "IoT Targeted"
        }

    def check_status(self):

        try:
            ip = self.target

            api_endpoint = f'{self.base_url}/check?ipAddress={ip}&maxAgeInDays={self.max_age}&verbose'
            response = requests.get(api_endpoint, headers=self.headers)

            if response.status_code == 429:
                return response.json()

            elif response.status_code == 200:
                response_data = response.json()
                abuseipdb_data = response_data.get('data')
                isp = abuseipdb_data['isp']
                total_reports = abuseipdb_data['totalReports']
                last_report_timestamp = abuseipdb_data['lastReportedAt']
                reports_list = abuseipdb_data['reports']
                reports_range = range(0,len(reports_list))

                if total_reports == 0:
                    return response_data
                else:
                    return reports_list[0]

        except ValueError:
            pass