#include "vrpn_Connection.h"
#include <cstdio>
#include <unistd.h>
int main(int argc, char** argv){
  if(argc<2){ printf("usage: %s host[:port]\n", argv[0]); return 1; }
  const char* host = argv[1];
  vrpn_Connection* c = vrpn_get_connection_by_name(host);
  printf("Connecting to %s ...\n", host);
  for(int i=0;i<300;i++){ c->mainloop(); usleep(10000); } // ~3s pump
  printf("connected=%d\n", c->connected());
  printf("Senders:\n");
  for(int i=0;;i++){ const char* n=c->sender_name(i); if(!n) break; printf("  [%d] %s\n",i,n); }
  printf("Message types:\n");
  for(int i=0;;i++){ const char* n=c->message_type_name(i); if(!n) break; printf("  [%d] %s\n",i,n); }
  return 0;
}
