// CloudFront Function (cloudfront-js-2.0, viewer-request) for the RFM video history gallery.
//
// Only /g/<id>/... is served. <id> is the 32-hex-char hash of the notification e-mail
// (HMAC-SHA256 with a per-account salt, see gallery.py). The KeyValueStore maps <id> to
// sha256hex(<id> + password); the viewer must send that password with HTTP Basic auth
// (any user name). Unknown <id> and wrong password both answer 401, so the response does not
// tell which ids exist. /g/<id> redirects to /g/<id>/ and /g/<id>/ serves /g/<id>/index.html.
import crypto from 'crypto';
import cf from 'cloudfront';

const kvs = cf.kvs();
const PATH = /^\/g\/([0-9a-f]{32})(\/.*)?$/;

function deny() {
  return {
    statusCode: 401,
    statusDescription: 'Unauthorized',
    headers: {
      'www-authenticate': { value: 'Basic realm="RFM video history", charset="UTF-8"' },
      'cache-control': { value: 'no-store' },
    },
  };
}

async function handler(event) {
  const request = event.request;
  const m = PATH.exec(request.uri);
  if (!m) {
    return { statusCode: 404, statusDescription: 'Not Found' };
  }
  const id = m[1];
  if (!m[2]) {
    return { statusCode: 302, statusDescription: 'Found', headers: { location: { value: '/g/' + id + '/' } } };
  }
  const auth = request.headers.authorization;
  if (!auth || auth.value.slice(0, 6).toLowerCase() !== 'basic ') {
    return deny();
  }
  let decoded;
  try {
    decoded = atob(auth.value.slice(6).trim());
  } catch (e) {
    return deny();
  }
  const sep = decoded.indexOf(':');
  if (sep < 0) {
    return deny();
  }
  let expected;
  try {
    expected = await kvs.get(id);
  } catch (e) {
    return deny(); // no such id
  }
  const digest = crypto.createHash('sha256').update(id + decoded.slice(sep + 1)).digest('hex');
  if (digest !== expected) {
    return deny();
  }
  if (m[2] === '/') {
    request.uri = '/g/' + id + '/index.html';
  }
  return request;
}
