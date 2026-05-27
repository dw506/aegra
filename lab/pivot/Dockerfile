FROM alpine:3.20

RUN apk add --no-cache \
        bash \
        bind-tools \
        busybox-extras \
        curl \
        iproute2 \
        iputils \
        nmap \
        openssh-server \
        postgresql-client \
    && adduser -D -s /bin/bash pivot \
    && echo "pivot:pivotpass" | chpasswd \
    && ssh-keygen -A \
    && sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?AllowTcpForwarding .*/AllowTcpForwarding yes/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PermitTunnel .*/PermitTunnel yes/' /etc/ssh/sshd_config \
    && printf '\nPermitRootLogin no\nAllowUsers pivot\nGatewayPorts no\nX11Forwarding no\n' >> /etc/ssh/sshd_config

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D", "-e"]
