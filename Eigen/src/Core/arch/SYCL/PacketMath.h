// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*****************************************************************
 * PacketMath.h
 *
 * \brief:
 *  PacketMath
 *
 *****************************************************************/

#ifndef EIGEN_PACKET_MATH_SYCL_H
#define EIGEN_PACKET_MATH_SYCL_H
#include <type_traits>
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {
#ifdef SYCL_DEVICE_ONLY

#define SYCL_PLOADT_RO(address_space_target)                                 \
  template <typename packet_type, int Alignment>                             \
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE packet_type ploadt_ro(               \
      typename hipsycl::sycl::multi_ptr<                                          \
          const typename unpacket_traits<packet_type>::type,                 \
          hipsycl::sycl::access::address_space::address_space_target>::pointer_t  \
          from) {                                                            \
    typedef typename unpacket_traits<packet_type>::type scalar;              \
    typedef hipsycl::sycl::multi_ptr<                                             \
        scalar, hipsycl::sycl::access::address_space::address_space_target>       \
        multi_ptr;                                                           \
    auto res = packet_type(                                                  \
        static_cast<typename unpacket_traits<packet_type>::type>(0));        \
    res.load(0, multi_ptr(const_cast<typename multi_ptr::pointer_t>(from))); \
    return res;                                                              \
  }

SYCL_PLOADT_RO(global_space)
SYCL_PLOADT_RO(local_space)
#undef SYCL_PLOADT_RO
#endif

template <typename packet_type, int Alignment, typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE packet_type
ploadt_ro(const Eigen::TensorSycl::internal::RangeAccess<
          hipsycl::sycl::access::mode::read_write, T>& from) {
  return ploadt_ro<packet_type, Alignment>(from.get_pointer());
}

#ifdef SYCL_DEVICE_ONLY
#define SYCL_PLOAD(address_space_target, Alignment, AlignedType)            \
  template <typename packet_type>                                           \
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE packet_type pload##AlignedType(     \
      typename hipsycl::sycl::multi_ptr<                                         \
          const typename unpacket_traits<packet_type>::type,                \
          hipsycl::sycl::access::address_space::address_space_target>::pointer_t \
          from) {                                                           \
    return ploadt_ro<packet_type, Alignment>(from);                         \
  }

// global space
SYCL_PLOAD(global_space, Unaligned, u)
SYCL_PLOAD(global_space, Aligned, )
// local space
SYCL_PLOAD(local_space, Unaligned, u)
SYCL_PLOAD(local_space, Aligned, )

#undef SYCL_PLOAD
#endif

#define SYCL_PLOAD(Alignment, AlignedType)                              \
  template <typename packet_type>                                       \
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE packet_type pload##AlignedType( \
      const Eigen::TensorSycl::internal::RangeAccess<                   \
          hipsycl::sycl::access::mode::read_write,                           \
          typename unpacket_traits<packet_type>::type>                  \
          from) {                                                       \
    return ploadt_ro<packet_type, Alignment>(from);                     \
  }
SYCL_PLOAD(Unaligned, u)
SYCL_PLOAD(Aligned, )
#undef SYCL_PLOAD

#ifdef SYCL_DEVICE_ONLY
/** \internal \returns a packet version of \a *from.
 * The pointer \a from must be aligned on a \a Alignment bytes boundary. */
#define SYCL_PLOADT(address_space_target)                                   \
  template <typename packet_type, int Alignment>                            \
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE packet_type ploadt(                 \
      typename hipsycl::sycl::multi_ptr<                                         \
          const typename unpacket_traits<packet_type>::type,                \
          hipsycl::sycl::access::address_space::address_space_target>::pointer_t \
          from) {                                                           \
    if (Alignment >= unpacket_traits<packet_type>::alignment)               \
      return pload<packet_type>(from);                                      \
    else                                                                    \
      return ploadu<packet_type>(from);                                     \
  }

// global space
SYCL_PLOADT(global_space)
// local space
SYCL_PLOADT(local_space)
#undef SYCL_PLOADT
#endif

template <typename packet_type, int Alignment>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE packet_type
ploadt(const Eigen::TensorSycl::internal::RangeAccess<
       hipsycl::sycl::access::mode::read_write,
       typename unpacket_traits<packet_type>::type>& from) {
  return ploadt<packet_type, Alignment>(from.get_pointer());
}
#ifdef SYCL_DEVICE_ONLY

// private_space
#define SYCL_PLOADT_RO_SPECIAL(packet_type, Alignment)                 \
  template <>                                                          \
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE packet_type                    \
  ploadt_ro<packet_type, Alignment>(                                   \
    const typename unpacket_traits<packet_type>::type* from) {       \
    typedef typename unpacket_traits<packet_type>::type scalar;        \
    typedef hipsycl::sycl::multi_ptr<scalar, hipsycl::sycl::access::address_space::private_space>  multi_ptr; \
    auto res = packet_type(static_cast<scalar>(0));                    \
    res.template load<hipsycl::sycl::access::address_space::private_space>( \
	  0, multi_ptr(const_cast<scalar*>(from)) );                       \
    return res;                                                        \
  }

SYCL_PLOADT_RO_SPECIAL(hipsycl::sycl::float4, Aligned)
SYCL_PLOADT_RO_SPECIAL(hipsycl::sycl::double2, Aligned)
SYCL_PLOADT_RO_SPECIAL(hipsycl::sycl::float4, Unaligned)
SYCL_PLOADT_RO_SPECIAL(hipsycl::sycl::double2, Unaligned)

#define SYCL_PLOAD_SPECIAL(packet_type, alignment_type)                    \
  template <>                                                              \
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE packet_type pload##alignment_type( \
     const typename unpacket_traits<packet_type>::type* from) {           \
    typedef typename unpacket_traits<packet_type>::type scalar;            \
    typedef hipsycl::sycl::multi_ptr< scalar, hipsycl::sycl::access::address_space::private_space>  multi_ptr; \
    auto res = packet_type(static_cast<scalar>(0));                        \
    res.template load<hipsycl::sycl::access::address_space::private_space>(     \
        0, multi_ptr(const_cast<scalar*>(from)));                                     \
    return res;                                                            \
  }
SYCL_PLOAD_SPECIAL(hipsycl::sycl::float4, )
SYCL_PLOAD_SPECIAL(hipsycl::sycl::double2, )
SYCL_PLOAD_SPECIAL(hipsycl::sycl::float4, u)
SYCL_PLOAD_SPECIAL(hipsycl::sycl::double2, u)

#undef SYCL_PLOAD_SPECIAL

#define SYCL_PSTORE(scalar, packet_type, address_space_target, alignment)   \
  template <>                                                               \
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void pstore##alignment(             \
      typename hipsycl::sycl::multi_ptr<                                         \
          scalar,                                                           \
          hipsycl::sycl::access::address_space::address_space_target>::pointer_t \
          to,                                                               \
      const packet_type& from) {                                            \
    typedef hipsycl::sycl::multi_ptr<                                            \
        scalar, hipsycl::sycl::access::address_space::address_space_target>      \
        multi_ptr;                                                          \
    from.store(0, multi_ptr(to));                                           \
  }

// global space
SYCL_PSTORE(float, hipsycl::sycl::float4, global_space, )
SYCL_PSTORE(float, hipsycl::sycl::float4, global_space, u)
SYCL_PSTORE(double, hipsycl::sycl::double2, global_space, )
SYCL_PSTORE(double, hipsycl::sycl::double2, global_space, u)
//SYCL_PSTORE(float, hipsycl::sycl::float4, local_space, )
//SYCL_PSTORE(float, hipsycl::sycl::float4, local_space, u)
//SYCL_PSTORE(double, hipsycl::sycl::double2, local_space, )
//SYCL_PSTORE(double, hipsycl::sycl::double2, local_space, u)

//SYCL_PSTORE(float, hipsycl::sycl::float4, private_space, )
//SYCL_PSTORE(float, hipsycl::sycl::float4, private_space, u)
//SYCL_PSTORE(double, hipsycl::sycl::double2, private_space, )
//SYCL_PSTORE(double, hipsycl::sycl::double2, private_space, u)
#undef SYCL_PSTORE

#define SYCL_PSTORE_T(address_space_target)                                 \
  template <typename scalar, typename packet_type, int Alignment>           \
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void pstoret(                       \
      typename hipsycl::sycl::multi_ptr<                                         \
          scalar,                                                           \
          hipsycl::sycl::access::address_space::address_space_target>::pointer_t \
          to,                                                               \
      const packet_type& from) {                                            \
    if (Alignment)                                                          \
      pstore(to, from);                                                     \
    else                                                                    \
      pstoreu(to, from);                                                    \
  }

SYCL_PSTORE_T(global_space)

SYCL_PSTORE_T(local_space)

#undef SYCL_PSTORE_T

#define SYCL_PSET1(packet_type)                                         \
  template <>                                                           \
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE packet_type pset1<packet_type>( \
      const typename unpacket_traits<packet_type>::type& from) {        \
    return packet_type(from);                                           \
  }

// global space
SYCL_PSET1(hipsycl::sycl::float4)
SYCL_PSET1(hipsycl::sycl::double2)

#undef SYCL_PSET1

template <typename packet_type>
struct get_base_packet {
  template <typename sycl_multi_pointer>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE packet_type
  get_ploaddup(sycl_multi_pointer) {}

  template <typename sycl_multi_pointer>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE packet_type
  get_pgather(sycl_multi_pointer, Index) {}
};

template <>
struct get_base_packet<hipsycl::sycl::float4> {
  template <typename sycl_multi_pointer>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE hipsycl::sycl::float4 get_ploaddup(
      sycl_multi_pointer from) {
    return hipsycl::sycl::float4(from[0], from[0], from[1], from[1]);
  }
  template <typename sycl_multi_pointer>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE hipsycl::sycl::float4 get_pgather(
      sycl_multi_pointer from, Index stride) {
    return hipsycl::sycl::float4(from[0 * stride], from[1 * stride],
                               from[2 * stride], from[3 * stride]);
  }

  template <typename sycl_multi_pointer>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void set_pscatter(
      sycl_multi_pointer to, const hipsycl::sycl::float4& from, Index stride) {
    auto tmp = stride;
    to[0] = from.x();
    to[tmp] = from.y();
    to[tmp += stride] = from.z();
    to[tmp += stride] = from.w();
  }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE hipsycl::sycl::float4 set_plset(
      const float& a) {
    return hipsycl::sycl::float4(static_cast<float>(a), static_cast<float>(a + 1),
                               static_cast<float>(a + 2),
                               static_cast<float>(a + 3));
  }
};

template <>
struct get_base_packet<hipsycl::sycl::double2> {
  template <typename sycl_multi_pointer>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE hipsycl::sycl::double2
  get_ploaddup(const sycl_multi_pointer from) {
    return hipsycl::sycl::double2(from[0], from[0]);
  }

  template <typename sycl_multi_pointer, typename Index>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE hipsycl::sycl::double2 get_pgather(
      const sycl_multi_pointer from, Index stride) {
    return hipsycl::sycl::double2(from[0 * stride], from[1 * stride]);
  }

  template <typename sycl_multi_pointer>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void set_pscatter(
      sycl_multi_pointer to, const hipsycl::sycl::double2& from, Index stride) {
    to[0] = from.x();
    to[stride] = from.y();
  }

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE hipsycl::sycl::double2 set_plset(
      const double& a) {
    return hipsycl::sycl::double2(static_cast<double>(a),
                                static_cast<double>(a + 1));
  }
};

#define SYCL_PLOAD_DUP(address_space_target)                                \
  template <typename packet_type>                                           \
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE packet_type ploaddup(               \
      typename hipsycl::sycl::multi_ptr<                                         \
          const typename unpacket_traits<packet_type>::type,                \
          hipsycl::sycl::access::address_space::address_space_target>::pointer_t \
          from) {                                                           \
    return get_base_packet<packet_type>::get_ploaddup(from);                \
  }

// global space
SYCL_PLOAD_DUP(global_space)
// local_space
SYCL_PLOAD_DUP(local_space)
#undef SYCL_PLOAD_DUP

#define SYCL_PLOAD_DUP_SPECILIZE(packet_type)                              \
  template <>                                                              \
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE packet_type ploaddup<packet_type>( \
      const typename unpacket_traits<packet_type>::type* from) {           \
    return get_base_packet<packet_type>::get_ploaddup(from);               \
  }

SYCL_PLOAD_DUP_SPECILIZE(hipsycl::sycl::float4)
SYCL_PLOAD_DUP_SPECILIZE(hipsycl::sycl::double2)

#undef SYCL_PLOAD_DUP_SPECILIZE

#define SYCL_PLSET(packet_type)                                         \
  template <>                                                           \
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE packet_type plset<packet_type>( \
      const typename unpacket_traits<packet_type>::type& a) {           \
    return get_base_packet<packet_type>::set_plset(a);                  \
  }

SYCL_PLSET(hipsycl::sycl::float4)
SYCL_PLSET(hipsycl::sycl::double2)

#undef SYCL_PLSET

#define SYCL_PGATHER(address_space_target)                                  \
  template <typename Scalar, typename packet_type>                          \
  EIGEN_DEVICE_FUNC inline packet_type pgather(                             \
      typename hipsycl::sycl::multi_ptr<                                         \
          const typename unpacket_traits<packet_type>::type,                \
          hipsycl::sycl::access::address_space::address_space_target>::pointer_t \
          from,                                                             \
      Index stride) {                                                       \
    return get_base_packet<packet_type>::get_pgather(from, stride);         \
  }

// global space
SYCL_PGATHER(global_space)
// local space
SYCL_PGATHER(local_space)

#undef SYCL_PGATHER

#define SYCL_PGATHER_SPECILIZE(scalar, packet_type)                            \
  template <>                                                                  \
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE packet_type                            \
  pgather<scalar, packet_type>(                                                \
      const typename unpacket_traits<packet_type>::type* from, Index stride) { \
    return get_base_packet<packet_type>::get_pgather(from, stride);            \
  }

SYCL_PGATHER_SPECILIZE(float, hipsycl::sycl::float4)
SYCL_PGATHER_SPECILIZE(double, hipsycl::sycl::double2)

#undef SYCL_PGATHER_SPECILIZE

#define SYCL_PSCATTER(address_space_target)                                 \
  template <typename Scalar, typename packet_type>                          \
  EIGEN_DEVICE_FUNC inline void pscatter(                                   \
      typename hipsycl::sycl::multi_ptr<                                         \
          typename unpacket_traits<packet_type>::type,                      \
          hipsycl::sycl::access::address_space::address_space_target>::pointer_t \
          to,                                                               \
      const packet_type& from, Index stride) {                              \
    get_base_packet<packet_type>::set_pscatter(to, from, stride);           \
  }

// global space
SYCL_PSCATTER(global_space)
// local space
SYCL_PSCATTER(local_space)

#undef SYCL_PSCATTER

#define SYCL_PSCATTER_SPECILIZE(scalar, packet_type)                        \
  template <>                                                               \
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pscatter<scalar, packet_type>( \
      typename unpacket_traits<packet_type>::type * to,                     \
      const packet_type& from, Index stride) {                              \
    get_base_packet<packet_type>::set_pscatter(to, from, stride);           \
  }

SYCL_PSCATTER_SPECILIZE(float, hipsycl::sycl::float4)
SYCL_PSCATTER_SPECILIZE(double, hipsycl::sycl::double2)

#undef SYCL_PSCATTER_SPECILIZE

#define SYCL_PMAD(packet_type)                                            \
  template <>                                                             \
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE packet_type pmadd(                \
      const packet_type& a, const packet_type& b, const packet_type& c) { \
    return hipsycl::sycl::mad(a, b, c);                                        \
  }

SYCL_PMAD(hipsycl::sycl::float4)
SYCL_PMAD(hipsycl::sycl::double2)
#undef SYCL_PMAD

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float pfirst<hipsycl::sycl::float4>(
    const hipsycl::sycl::float4& a) {
  return a.x();
}
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE double pfirst<hipsycl::sycl::double2>(
    const hipsycl::sycl::double2& a) {
  return a.x();
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float predux<hipsycl::sycl::float4>(
    const hipsycl::sycl::float4& a) {
  return a.x() + a.y() + a.z() + a.w();
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE double predux<hipsycl::sycl::double2>(
    const hipsycl::sycl::double2& a) {
  return a.x() + a.y();
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float predux_max<hipsycl::sycl::float4>(
    const hipsycl::sycl::float4& a) {
  return hipsycl::sycl::fmax(hipsycl::sycl::fmax(a.x(), a.y()),
                        hipsycl::sycl::fmax(a.z(), a.w()));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE double predux_max<hipsycl::sycl::double2>(
    const hipsycl::sycl::double2& a) {
  return hipsycl::sycl::fmax(a.x(), a.y());
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float predux_min<hipsycl::sycl::float4>(
    const hipsycl::sycl::float4& a) {
  return hipsycl::sycl::fmin(hipsycl::sycl::fmin(a.x(), a.y()),
                        hipsycl::sycl::fmin(a.z(), a.w()));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE double predux_min<hipsycl::sycl::double2>(
    const hipsycl::sycl::double2& a) {
  return hipsycl::sycl::fmin(a.x(), a.y());
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float predux_mul<hipsycl::sycl::float4>(
    const hipsycl::sycl::float4& a) {
  return a.x() * a.y() * a.z() * a.w();
}
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE double predux_mul<hipsycl::sycl::double2>(
    const hipsycl::sycl::double2& a) {
  return a.x() * a.y();
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE hipsycl::sycl::float4
pabs<hipsycl::sycl::float4>(const hipsycl::sycl::float4& a) {
  return hipsycl::sycl::float4(hipsycl::sycl::fabs(a.x()), hipsycl::sycl::fabs(a.y()),
                             hipsycl::sycl::fabs(a.z()), hipsycl::sycl::fabs(a.w()));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE hipsycl::sycl::double2
pabs<hipsycl::sycl::double2>(const hipsycl::sycl::double2& a) {
  return hipsycl::sycl::double2(hipsycl::sycl::fabs(a.x()), hipsycl::sycl::fabs(a.y()));
}

template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet sycl_pcmp_le(const Packet &a,
                                                          const Packet &b) {
  return ((a <= b)
              .template convert<typename unpacket_traits<Packet>::type,
                                hipsycl::sycl::rounding_mode::automatic>());
}

template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet sycl_pcmp_lt(const Packet &a,
                                                          const Packet &b) {
  return ((a < b)
              .template convert<typename unpacket_traits<Packet>::type,
                                hipsycl::sycl::rounding_mode::automatic>());
}

template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet sycl_pcmp_eq(const Packet &a,
                                                          const Packet &b) {
  return ((a == b)
              .template convert<typename unpacket_traits<Packet>::type,
                                hipsycl::sycl::rounding_mode::automatic>());
}

#define SYCL_PCMP(OP, TYPE)                                                    \
  template <>                                                                  \
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE TYPE pcmp_##OP<TYPE>(const TYPE &a,    \
                                                             const TYPE &b) {  \
    return sycl_pcmp_##OP<TYPE>(a, b);                                         \
  }

SYCL_PCMP(le, hipsycl::sycl::float4)
SYCL_PCMP(lt, hipsycl::sycl::float4)
SYCL_PCMP(eq, hipsycl::sycl::float4)
SYCL_PCMP(le, hipsycl::sycl::double2)
SYCL_PCMP(lt, hipsycl::sycl::double2)
SYCL_PCMP(eq, hipsycl::sycl::double2)
#undef SYCL_PCMP

template <typename T> struct convert_to_integer;

template <> struct convert_to_integer<float> {
  using type = std::int32_t;
  using packet_type = hipsycl::sycl::int4;
};
template <> struct convert_to_integer<double> {
  using type = std::int64_t;
  using packet_type = hipsycl::sycl::long2;
};

template <typename PacketIn>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename convert_to_integer<
    typename unpacket_traits<PacketIn>::type>::packet_type
vector_as_int(const PacketIn &p) {
  return (
      p.template convert<typename convert_to_integer<
                             typename unpacket_traits<PacketIn>::type>::type,
                         hipsycl::sycl::rounding_mode::automatic>());
}

template <typename packetOut, typename PacketIn>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE packetOut
convert_vector(const PacketIn &p) {
  return (p.template convert<typename unpacket_traits<packetOut>::type,
                             hipsycl::sycl::rounding_mode::automatic>());
}

#define SYCL_PAND(TYPE)                                                        \
  template <>                                                                  \
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TYPE pand<TYPE>(const TYPE &a,         \
                                                        const TYPE &b) {       \
    return convert_vector<TYPE>(vector_as_int(a) & vector_as_int(b));          \
  }
SYCL_PAND(hipsycl::sycl::float4)
SYCL_PAND(hipsycl::sycl::double2)
#undef SYCL_PAND

#define SYCL_POR(TYPE)                                                         \
  template <>                                                                  \
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TYPE por<TYPE>(const TYPE &a,          \
                                                       const TYPE &b) {        \
    return convert_vector<TYPE>(vector_as_int(a) | vector_as_int(b));          \
  }

SYCL_POR(hipsycl::sycl::float4)
SYCL_POR(hipsycl::sycl::double2)
#undef SYCL_POR

#define SYCL_PXOR(TYPE)                                                        \
  template <>                                                                  \
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TYPE pxor<TYPE>(const TYPE &a,         \
                                                        const TYPE &b) {       \
    return convert_vector<TYPE>(vector_as_int(a) ^ vector_as_int(b));          \
  }

SYCL_PXOR(hipsycl::sycl::float4)
SYCL_PXOR(hipsycl::sycl::double2)
#undef SYCL_PXOR

#define SYCL_PANDNOT(TYPE)                                                     \
  template <>                                                                  \
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TYPE pandnot<TYPE>(const TYPE &a,      \
                                                           const TYPE &b) {    \
    return convert_vector<TYPE>(vector_as_int(a) & (~vector_as_int(b)));       \
  }
SYCL_PANDNOT(hipsycl::sycl::float4)
SYCL_PANDNOT(hipsycl::sycl::double2)
#undef SYCL_PANDNOT

EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void ptranspose(
    PacketBlock<hipsycl::sycl::float4, 4>& kernel) {
  float tmp = kernel.packet[0].y();
  kernel.packet[0].y() = kernel.packet[1].x();
  kernel.packet[1].x() = tmp;

  tmp = kernel.packet[0].z();
  kernel.packet[0].z() = kernel.packet[2].x();
  kernel.packet[2].x() = tmp;

  tmp = kernel.packet[0].w();
  kernel.packet[0].w() = kernel.packet[3].x();
  kernel.packet[3].x() = tmp;

  tmp = kernel.packet[1].z();
  kernel.packet[1].z() = kernel.packet[2].y();
  kernel.packet[2].y() = tmp;

  tmp = kernel.packet[1].w();
  kernel.packet[1].w() = kernel.packet[3].y();
  kernel.packet[3].y() = tmp;

  tmp = kernel.packet[2].w();
  kernel.packet[2].w() = kernel.packet[3].z();
  kernel.packet[3].z() = tmp;
}

EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void ptranspose(
    PacketBlock<hipsycl::sycl::double2, 2>& kernel) {
  double tmp = kernel.packet[0].y();
  kernel.packet[0].y() = kernel.packet[1].x();
  kernel.packet[1].x() = tmp;
}

//// geninteger select (geninteger a, geninteger b, igeninteger c)
//template <typename T, typename T2>
//T select(T a, T b, T2 c) {
//  return __invoke_select<T>(a, b, c);
//}

inline hipsycl::sycl::float4 select(const hipsycl::sycl::float4& thenVec,
                                        const hipsycl::sycl::float4& elseVec,
                                        const hipsycl::sycl::int4& condition) {
    // Use bitwise operations to perform conditional selection
    return hipsycl::sycl::float4{ condition.x() ? thenVec.x() : elseVec.x(),
                                   condition.y() ? thenVec.y() : elseVec.y(),
                                   condition.z() ? thenVec.z() : elseVec.z(),
                                   condition.w() ? thenVec.w() : elseVec.w() };
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE hipsycl::sycl::float4 pblend(
    const Selector<unpacket_traits<hipsycl::sycl::float4>::size>& ifPacket,
    const hipsycl::sycl::float4& thenPacket,
    const hipsycl::sycl::float4& elsePacket) {
  hipsycl::sycl::int4 condition(
      ifPacket.select[0] ? 0 : -1, ifPacket.select[1] ? 0 : -1,
      ifPacket.select[2] ? 0 : -1, ifPacket.select[3] ? 0 : -1);
  return select(thenPacket, elsePacket, condition);
}

//template <>
//inline hipsycl::sycl::double2 pblend(
//    const Selector<unpacket_traits<hipsycl::sycl::double2>::size>& ifPacket,
//    const hipsycl::sycl::double2& thenPacket,
//    const hipsycl::sycl::double2& elsePacket) {
//  hipsycl::sycl::long2 condition(ifPacket.select[0] ? 0 : -1,
//                               ifPacket.select[1] ? 0 : -1);
//  return select(thenPacket, elsePacket, condition);
//}
#endif  // SYCL_DEVICE_ONLY

#define SYCL_PSTORE(alignment)                                  \
  template <typename packet_type>                               \
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void pstore##alignment( \
      const Eigen::TensorSycl::internal::RangeAccess<           \
          hipsycl::sycl::access::mode::read_write,                   \
          typename unpacket_traits<packet_type>::type>& to,     \
      const packet_type& from) {                                \
    pstore##alignment(to.get_pointer(), from);                  \
  }

// global space
SYCL_PSTORE()
SYCL_PSTORE(u)

#undef SYCL_PSTORE

template <typename scalar, typename packet_type, int Alignment>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void pstoret(
    Eigen::TensorSycl::internal::RangeAccess<
        hipsycl::sycl::access::mode::read_write,
        typename unpacket_traits<packet_type>::type>
        to,
    const packet_type& from) {
  pstoret<scalar, packet_type, Alignment>(to.get_pointer(), from);
}

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_PACKET_MATH_SYCL_H
