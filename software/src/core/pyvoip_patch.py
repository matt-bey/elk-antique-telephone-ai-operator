"""
Monkey-patch pyVoIP to support 407 Proxy Authentication Required and
fix ACK handling for Callcentric's SBC.

Callcentric responds to SIP REGISTER and INVITE with 407 +
Proxy-Authenticate instead of 401 + WWW-Authenticate.  The digest
calculation is identical; only the header names differ:

    401: WWW-Authenticate  → Authorization
    407: Proxy-Authenticate → Proxy-Authorization

pyVoIP 1.6.8 has a TODO stub for 407 (SIP.py line 1840).  This module
patches seven things so registration, outbound calls, and audio work:

  1. SIPMessage header parser — recognize Proxy-Authenticate as auth header
  2. SIPClient.__register  — handle 407 the same way as 401
  3. SIPClient.gen_register — emit Proxy-Authorization when responding to 407
  4. SIPClient.invite — handle 407 on INVITE (call placement)
  5. VoIPPhone._callback_RESP_OK — fix ACK for 200 OK: use Contact URI as
     Request-URI and send to the SBC address from the Contact header
  6. SIPClient.bye — send BYE to the SBC (Contact address) not the registrar
  7. VoIPCall.answered — stop pyVoIP's RTPClients after SDP parsing and
     expose connection params for our custom RTP stream (rtp_stream.py)

Call ``apply()`` once before creating any VoIPPhone instance.
"""

import hashlib
import logging
import re
import select

logger = logging.getLogger(__name__)

_applied = False


def apply() -> None:
    """Apply the 407 patches to the imported pyVoIP module."""
    global _applied
    if _applied:
        return

    try:
        from pyVoIP.SIP import SIPClient, SIPMessage, SIPStatus
        from pyVoIP.VoIP import PhoneStatus, VoIPCall
        from pyVoIP import DEBUG as _dbg_flag
        from pyVoIP.SIP import debug
    except ImportError:
        logger.debug("pyVoIP not available — skipping 407 patch")
        return

    # ------------------------------------------------------------------
    # 1. Patch SIPMessage.parse_header to recognise Proxy-Authenticate
    # ------------------------------------------------------------------
    _orig_parse_header = SIPMessage.parse_header

    def _patched_parse_header(self, header: str, data: str) -> None:
        if header == "Proxy-Authenticate":
            # Parse exactly like WWW-Authenticate
            data = data.replace("Digest ", "")
            row_data = self.auth_match.findall(data)
            header_data = {}
            for var, val in row_data:
                header_data[var] = val.strip('"')
            self.headers[header] = header_data
            self.authentication = header_data
        else:
            _orig_parse_header(self, header, data)

    SIPMessage.parse_header = _patched_parse_header

    # ------------------------------------------------------------------
    # 2 & 3. Patch __register to handle 407 and gen_register to emit
    #         the correct Authorization / Proxy-Authorization header.
    # ------------------------------------------------------------------
    _orig_gen_register = SIPClient.gen_register

    def _patched_gen_register(self, request, deregister=False, proxy=False):
        """Generate an authenticated REGISTER request.

        When *proxy* is True the header is ``Proxy-Authorization`` instead
        of ``Authorization`` (required for 407 responses).
        """
        result = _orig_gen_register(self, request, deregister=deregister)
        if proxy:
            result = result.replace("Authorization: Digest", "Proxy-Authorization: Digest", 1)
        return result

    SIPClient.gen_register = _patched_gen_register

    # The private name-mangled method __register is stored as
    # _SIPClient__register on the class.
    _orig_register = getattr(SIPClient, "_SIPClient__register")

    def _patched__register(self) -> bool:
        """Registration flow with 407 Proxy-Auth support."""
        self.phone._status = PhoneStatus.REGISTERING
        firstRequest = self.gen_first_response()
        self.out.sendto(firstRequest.encode("utf8"), (self.server, self.port))

        self.out.setblocking(False)

        ready = select.select([self.out], [], [], self.register_timeout)
        if ready[0]:
            resp = self.s.recv(8192)
        else:
            from pyVoIP.SIP import TimeoutError as SIPTimeout
            raise SIPTimeout("Registering on SIP Server timed out")

        response = SIPMessage(resp)
        response = self.trying_timeout_check(response)
        first_response = response

        # --- 400 Bad Request ---
        if response.status == SIPStatus(400):
            self._handle_bad_request()

        # --- 401 Unauthorized (original pyVoIP flow) ---
        if response.status == SIPStatus(401):
            return self._handle_auth_challenge(
                response, first_response, firstRequest, proxy=False
            )

        # --- 407 Proxy Authentication Required (NEW) ---
        if response.status == SIPStatus(407):
            return self._handle_auth_challenge(
                response, first_response, firstRequest, proxy=True
            )

        # --- Other statuses (original pyVoIP flow) ---
        if response.status not in [
            SIPStatus(400), SIPStatus(401), SIPStatus(407),
        ]:
            if response.status == SIPStatus(500):
                from pyVoIP.SIP import RetryRequiredError
                raise RetryRequiredError("Response SIP status of 500")
            else:
                self.parse_message(response)

        debug(response.summary())
        debug(response.raw)

        if response.status == SIPStatus.OK:
            return True
        else:
            from pyVoIP.SIP import InvalidAccountInfoError
            raise InvalidAccountInfoError(
                f"Invalid Username or Password for SIP server "
                f"{self.server}:{self.myPort}"
            )

    def _handle_auth_challenge(self, response, first_response, firstRequest, proxy=False):
        """Shared handler for 401 and 407 auth challenges."""
        from pyVoIP.SIP import InvalidAccountInfoError

        auth_type = "Proxy" if proxy else "WWW"
        regRequest = self.gen_register(response, proxy=proxy)
        self.out.sendto(regRequest.encode("utf8"), (self.server, self.port))

        ready = select.select([self.s], [], [], self.register_timeout)
        if ready[0]:
            resp = self.s.recv(8192)
            response = SIPMessage(resp)
            response = self.trying_timeout_check(response)

            if response.status in (SIPStatus(401), SIPStatus(407)):
                debug("=" * 50)
                debug(f"{auth_type}-Auth failed, SIP Message Log:\n")
                debug("SENT")
                debug(firstRequest)
                debug("\nRECEIVED")
                debug(first_response.summary())
                debug("\nSENT (DO NOT SHARE THIS PACKET)")
                debug(regRequest)
                debug("\nRECEIVED")
                debug(response.summary())
                debug("=" * 50)
                raise InvalidAccountInfoError(
                    f"Invalid Username or Password for SIP server "
                    f"{self.server}:{self.myPort}"
                )
            elif response.status == SIPStatus(400):
                self._handle_bad_request()

            debug(response.summary())
            debug(response.raw)

            if response.status == SIPStatus.OK:
                return True
            else:
                raise InvalidAccountInfoError(
                    f"Invalid Username or Password for SIP server "
                    f"{self.server}:{self.myPort}"
                )
        else:
            from pyVoIP.SIP import TimeoutError as SIPTimeout
            raise SIPTimeout("Registering on SIP Server timed out")

    # ------------------------------------------------------------------
    # 4. Patch invite() to handle 407 on INVITE (call placement)
    # ------------------------------------------------------------------
    _orig_invite = SIPClient.invite

    def _patched_invite(self, number, ms, sendtype):
        """INVITE flow with 407 Proxy-Auth support.

        The original invite() only breaks its response loop on 401, 100,
        or 180.  Callcentric sends 407 for INVITE too, so we add it to
        the loop condition and generate Proxy-Authorization when needed.
        """
        branch = "z9hG4bK" + self.gen_call_id()[0:25]
        call_id = self.gen_call_id()
        sess_id = self.sessID.next()
        invite = self.gen_invite(
            number, str(sess_id), ms, sendtype, branch, call_id
        )
        with self.recvLock:
            self.out.sendto(invite.encode("utf8"), (self.server, self.port))
            debug("Invited")
            response = SIPMessage(self.s.recv(8192))

            # Loop until we get a status we can act on — added 407
            while (
                response.status != SIPStatus(401)
                and response.status != SIPStatus(407)
                and response.status != SIPStatus(100)
                and response.status != SIPStatus(180)
            ) or response.headers["Call-ID"] != call_id:
                if not self.NSD:
                    break
                self.parse_message(response)
                response = SIPMessage(self.s.recv(8192))

            # 100 Trying or 180 Ringing — call is progressing
            if response.status in (SIPStatus(100), SIPStatus(180)):
                return SIPMessage(invite.encode("utf8")), call_id, sess_id

            # 401 or 407 — need to authenticate the INVITE
            debug(f"Received Response: {response.summary()}")
            ack = self.gen_ack(response)
            self.out.sendto(ack.encode("utf8"), (self.server, self.port))
            debug("Acknowledged")

            nonce = response.authentication["nonce"]
            realm = response.authentication["realm"]

            # The digest URI MUST match the INVITE Request-URI exactly.
            # gen_authorization() hardcodes "sip:{server};transport=UDP"
            # which is wrong for INVITE — compute the digest manually.
            invite_uri = f"sip:{number}@{self.server}"
            ha1 = hashlib.md5(
                f"{self.username}:{realm}:{self.password}".encode("utf8")
            ).hexdigest()
            ha2 = hashlib.md5(
                f"INVITE:{invite_uri}".encode("utf8")
            ).hexdigest()
            response_hash = hashlib.md5(
                f"{ha1}:{nonce}:{ha2}".encode("utf8")
            ).hexdigest()

            # Use Proxy-Authorization for 407, Authorization for 401
            if response.status == SIPStatus(407):
                auth_header = "Proxy-Authorization"
            else:
                auth_header = "Authorization"

            auth = (
                f'{auth_header}: Digest username="{self.username}",'
                f'realm="{realm}",nonce="{nonce}",'
                f'uri="{invite_uri}",'
                f'response="{response_hash}",'
                f'algorithm=MD5\r\n'
            )

            invite = self.gen_invite(
                number, str(sess_id), ms, sendtype, branch, call_id
            )
            invite = invite.replace(
                "\r\nContent-Length", f"\r\n{auth}Content-Length"
            )

            self.out.sendto(invite.encode("utf8"), (self.server, self.port))
            return SIPMessage(invite.encode("utf8")), call_id, sess_id

    SIPClient.invite = _patched_invite

    # ------------------------------------------------------------------
    # 5. Patch VoIPPhone._callback_RESP_OK to fix ACK for 200 OK
    # ------------------------------------------------------------------
    # pyVoIP's ACK has two problems with Callcentric:
    #   a) Request-URI uses the To header, but RFC 3261 requires the
    #      Contact URI from the 200 OK for 2xx ACKs.
    #   b) ACK is sent to (self.server, self.port) — the registrar —
    #      but the SBC may be at a different address (Contact header).
    # Both cause the SBC to never receive/accept the ACK, so it
    # retransmits 200 OK and eventually sends BYE.
    from pyVoIP.VoIP import VoIPPhone

    _orig_callback_resp_ok = VoIPPhone._callback_RESP_OK

    def _patched_callback_resp_ok(self, request):
        """Handle 200 OK with a fully corrected ACK.

        pyVoIP's gen_ack() has three issues that cause Callcentric's SBC
        to reject the ACK and retransmit 200 OK until it gives up:
          a) Request-URI uses the To header — should be the Contact URI
          b) To tag is a new random value — must match the 200 OK's To tag
          c) ACK sent to registrar — should go to the SBC (Contact host)
        We build the ACK from scratch to fix all three.
        """
        debug("OK recieved")
        call_id = request.headers["Call-ID"]
        if call_id not in self.calls:
            debug("Unknown/No call")
            return

        self.calls[call_id].answered(request)
        debug("Answered")

        # --- Extract Contact URI and host for destination ---
        contact_uri = None
        contact_host = None
        contact_port = 5060
        if "Contact" in request.headers:
            raw_contact = request.headers["Contact"]
            if isinstance(raw_contact, str):
                m = re.search(r'<(sip:[^>]+)>', raw_contact)
                if m:
                    contact_uri = m.group(1)
                else:
                    contact_uri = raw_contact.strip("<>")
            elif isinstance(raw_contact, dict) and "raw" in raw_contact:
                contact_uri = raw_contact["raw"].strip("<>")

            if contact_uri:
                host_match = re.search(r'@([^:;>]+)(?::(\d+))?', contact_uri)
                if host_match:
                    contact_host = host_match.group(1)
                    if host_match.group(2):
                        contact_port = int(host_match.group(2))

        # --- Build ACK from scratch (RFC 3261 §13.2.2.4) ---
        # Request-URI = remote target (Contact from 200 OK)
        req_uri = contact_uri or request.headers["To"]["raw"].strip("<>")

        # Via: new branch for 2xx ACK (separate transaction)
        import uuid
        branch = "z9hG4bK" + uuid.uuid4().hex[:25]
        via = (
            f"Via: SIP/2.0/UDP {self.sip.myIP}:{self.sip.myPort}"
            f";branch={branch};rport\r\n"
        )

        # From: same as INVITE (our side) — use tag from tagLibrary
        from_tag = self.sip.tagLibrary.get(call_id, self.sip.gen_tag())
        from_hdr = (
            f"From: {request.headers['From']['raw']};tag={from_tag}\r\n"
        )

        # To: MUST match the 200 OK's To header exactly (including tag)
        to_raw = request.headers["To"]["raw"]
        to_tag = request.headers["To"].get("tag", "")
        if to_tag:
            to_hdr = f"To: {to_raw};tag={to_tag}\r\n"
        else:
            to_hdr = f"To: {to_raw}\r\n"

        cseq = request.headers["CSeq"]["check"]

        ack = (
            f"ACK {req_uri} SIP/2.0\r\n"
            f"{via}"
            f"Max-Forwards: 70\r\n"
            f"{to_hdr}"
            f"{from_hdr}"
            f"Call-ID: {call_id}\r\n"
            f"CSeq: {cseq} ACK\r\n"
            f"Content-Length: 0\r\n\r\n"
        )

        # Send ACK to the SBC address (Contact), falling back to registrar
        dest_host = contact_host or self.sip.server
        dest_port = contact_port
        self.sip.out.sendto(ack.encode("utf8"), (dest_host, dest_port))
        logger.debug(
            f"ACK sent to {dest_host}:{dest_port} "
            f"(uri={req_uri}, to_tag={to_tag})"
        )

    VoIPPhone._callback_RESP_OK = _patched_callback_resp_ok

    # ------------------------------------------------------------------
    # 6. Patch SIPClient.bye to send BYE to the SBC (Contact address)
    # ------------------------------------------------------------------
    # pyVoIP has a TODO: "Handle bye to server vs. bye to connected client"
    # and always sends to (self.server, self.port).  The SBC that handles
    # the dialog is at the Contact address from the 200 OK, not the
    # registrar.  Sending to the registrar returns 404 Not Found.
    _orig_bye = SIPClient.bye

    def _patched_bye(self, request):
        """Send BYE to the SBC's Contact address, not the registrar."""
        message = self.gen_bye(request)

        # Extract host from the Contact URI in the request (which was
        # updated from the 200 OK by VoIPCall.answered()).
        dest_host = self.server
        dest_port = self.port
        contact = request.headers.get("Contact", "")
        if contact:
            if isinstance(contact, dict):
                contact = contact.get("raw", "")
            host_match = re.search(r'@([^:;>]+)(?::(\d+))?', contact)
            if host_match:
                dest_host = host_match.group(1)
                if host_match.group(2):
                    dest_port = int(host_match.group(2))

        self.out.sendto(message.encode("utf8"), (dest_host, dest_port))
        logger.debug(f"BYE sent to {dest_host}:{dest_port}")

    SIPClient.bye = _patched_bye

    # ------------------------------------------------------------------
    # 7. Patch VoIPCall.answered() to stop pyVoIP RTP and expose params
    # ------------------------------------------------------------------
    # After pyVoIP parses the SDP and creates RTPClients (binding sockets,
    # starting send/recv threads), we immediately stop them and stash the
    # connection parameters.  Our custom RTPStream (rtp_stream.py) takes
    # over with direct int16→µ-law encoding for better audio quality.
    _orig_answered = VoIPCall.answered

    def _patched_answered(self, request):
        """Let pyVoIP parse SDP and create RTPClients, then stop them."""
        _orig_answered(self, request)

        self._rtp_params = []
        for rtp_client in self.RTPClients:
            self._rtp_params.append({
                'local_ip': rtp_client.inIP,
                'local_port': rtp_client.inPort,
                'remote_ip': rtp_client.outIP,
                'remote_port': rtp_client.outPort,
                'ssrc': rtp_client.outSSRC,
            })
            rtp_client.stop()

        logger.debug(f"Stopped pyVoIP RTPClients; params: {self._rtp_params}")

    VoIPCall.answered = _patched_answered

    # Install the registration patches
    setattr(SIPClient, "_SIPClient__register", _patched__register)
    SIPClient._handle_auth_challenge = _handle_auth_challenge

    _applied = True
    logger.info("pyVoIP patched: 407 Proxy-Auth + ACK/BYE fix enabled")
